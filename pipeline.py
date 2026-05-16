import torch
import tensorrt as trt
from diffusers import StableDiffusionPipeline, AutoencoderTiny, LCMScheduler

class StableTRT10Engine:
    def __init__(self, path, embed_dim=768):
        self.embed_dim = embed_dim
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.s_buf   = torch.empty((1, 4, 64, 64),  dtype=torch.float32, device="cuda")
        self.t_buf   = torch.empty((1,),             dtype=torch.float32, device="cuda")
        self.e_buf   = torch.empty((1, 77, embed_dim), dtype=torch.float16, device="cuda")
        self.out_buf = torch.empty((1, 4, 64, 64),   dtype=torch.float16, device="cuda")

    def __call__(self, latent_model_input, timestep, encoder_hidden_states):
        self.context.set_input_shape("sample",                 (1, 4, 64, 64))
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
    def __init__(self, model_path, engine_path, latent_scale=0.18215):
        self.latent_scale = latent_scale
        
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

        print("Loading TAESD...")
        self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to("cuda")

        print("Loading TensorRT Engine...")
        self.unet_engine = StableTRT10Engine(engine_path, embed_dim=768)
        
        self.shared_noise = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16)
        self.prev_x0 = None

    def get_embeds(self, prompt):
        tokens = self.pipe.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).to("cuda")
        return self.pipe.text_encoder(tokens.input_ids)[0].half()

    def step(self, input_image, embeds, strength, delta):
        with torch.no_grad():
            # input_image should be (1, 3, 512, 512) normalized to [-1, 1]
            init_latents = self.vae.encode(input_image).latents * self.latent_scale

            idx = max(0, min(int((1.0 - strength) * 3), 3))
            t = self.pipe.scheduler.timesteps[idx]

            noised_latents = self.pipe.scheduler.add_noise(init_latents, self.shared_noise, t.unsqueeze(0))
            latent_model_input = self.pipe.scheduler.scale_model_input(noised_latents, t)

            model_output = self.unet_engine(
                latent_model_input.float(),
                t.unsqueeze(0).float(),
                embeds.half()
            ).half()

            out = self.pipe.scheduler.step(model_output, t, noised_latents)
            x0_pred = out.denoised if hasattr(out, 'denoised') else out.prev_sample

            if self.prev_x0 is not None:
                x0_pred = torch.lerp(x0_pred, self.prev_x0, delta)
            self.prev_x0 = x0_pred.detach()

            decoded = self.vae.decode(x0_pred / self.latent_scale).sample
            output_image = (decoded / 2 + 0.5).clamp(0, 1)
            
            return output_image[0].permute(1, 2, 0).cpu().numpy()
