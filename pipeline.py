import torch
import tensorrt as trt
from diffusers import StableDiffusionPipeline, AutoencoderTiny, LCMScheduler


class StableTRT10Engine:
    def __init__(self, path, latent_h=64, latent_w=64, embed_dim=768):
        self.embed_dim = embed_dim
        self.latent_h = latent_h
        self.latent_w = latent_w
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.s_buf   = torch.empty((1, 4, latent_h, latent_w), dtype=torch.float32, device="cuda")
        self.t_buf   = torch.empty((1,),                       dtype=torch.float32, device="cuda")
        self.e_buf   = torch.empty((1, 77, embed_dim),         dtype=torch.float16, device="cuda")
        self.out_buf = torch.empty((1, 4, latent_h, latent_w), dtype=torch.float16, device="cuda")

    def __call__(self, latent_model_input, timestep, encoder_hidden_states):
        self.context.set_input_shape("sample",                 (1, 4, self.latent_h, self.latent_w))
        self.context.set_input_shape("timestep",               (1,))
        self.context.set_input_shape("encoder_hidden_states",  (1, 77, self.embed_dim))
        self.s_buf.copy_(latent_model_input.float())
        self.t_buf.copy_(timestep.float())
        self.e_buf.copy_(encoder_hidden_states.half())
        self.context.set_tensor_address("sample",                self.s_buf.data_ptr())
        self.context.set_tensor_address("timestep",              self.t_buf.data_ptr())
        self.context.set_tensor_address("encoder_hidden_states", self.e_buf.data_ptr())
        self.context.set_tensor_address("latent",                self.out_buf.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out_buf


class AIPipeline:
    def __init__(self, model_path, engine_path, width=512, height=512, latent_scale=0.18215):
        self.latent_scale = latent_scale
        self.latent_h = height // 8
        self.latent_w = width // 8

        print("Loading Stable Diffusion Pipeline...")
        self.pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            use_safetensors=True,
        ).to("cuda")

        print("Loading LCM LoRA...")
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self.pipe.fuse_lora()
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.set_timesteps(4, device="cuda")

        # Full HyperVAE for both encode and decode. TAESD on encode caused
        # subtle spatial distortion that washed out geometric warps when going
        # back through the full VAE on decode — a mismatch in their latent
        # spaces. Worth ~5ms per frame to keep the warp legible.
        self.vae = self.pipe.vae

        print("Loading TensorRT Engine...")
        self.unet_engine = StableTRT10Engine(
            engine_path, latent_h=self.latent_h, latent_w=self.latent_w, embed_dim=768,
        )

        self.shared_noise = torch.randn(
            (1, 4, self.latent_h, self.latent_w), device="cuda", dtype=torch.float16,
        )
        self.prev_output = None

    def get_embeds(self, prompt):
        tokens = self.pipe.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).to("cuda")
        return self.pipe.text_encoder(tokens.input_ids)[0].half()

    def step(self, input_image, embeds, strength, delta):
        """
        input_image: (1, 3, H, W) float16 CUDA tensor in [0, 1].
        Returns:     (1, 3, H, W) float16 CUDA tensor in [0, 1].
        """
        with torch.no_grad():
            x = input_image * 2.0 - 1.0
            init_latents = self.vae.encode(x).latent_dist.mode() * self.latent_scale

            # Fixed mid-range noise level — strength is handled in image space below.
            t = self.pipe.scheduler.timesteps[2]
            noised_latents = self.pipe.scheduler.add_noise(init_latents, self.shared_noise, t.unsqueeze(0))
            latent_model_input = self.pipe.scheduler.scale_model_input(noised_latents, t)

            model_output = self.unet_engine(
                latent_model_input.float(),
                t.unsqueeze(0).float(),
                embeds.half(),
            ).half()

            out = self.pipe.scheduler.step(model_output, t, noised_latents)
            x0_pred = out.denoised if hasattr(out, 'denoised') else out.prev_sample

            decoded = self.vae.decode(x0_pred / self.latent_scale).sample
            ai_image = (decoded / 2 + 0.5).clamp(0, 1)

            # Blend in IMAGE space — linear in pixels, so the spatial warp from
            # the visualizer is preserved at (1-strength) weight directly.
            output = torch.lerp(input_image, ai_image, strength)

            # Temporal smooth also in image space.
            if self.prev_output is not None:
                output = torch.lerp(output, self.prev_output, delta)
            self.prev_output = output.detach()

            return output
