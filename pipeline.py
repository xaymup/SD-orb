import gc

import torch
import tensorrt as trt
from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    LCMScheduler,
    StableDiffusionPipeline,
)


class StableTRT10Engine:
    def __init__(self, path, latent_h=64, latent_w=64, embed_dim=768):
        self.embed_dim = embed_dim
        self.latent_h = latent_h
        self.latent_w = latent_w
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        self.s_buf   = torch.empty((1, 4, latent_h, latent_w), dtype=torch.float32, device="cuda")
        self.t_buf   = torch.empty((1,),                       dtype=torch.float32, device="cuda")
        self.e_buf   = torch.empty((1, 77, embed_dim),         dtype=torch.float16, device="cuda")
        self.out_buf = torch.empty((1, 4, latent_h, latent_w), dtype=torch.float16, device="cuda")
        self.engine = None
        self.context = None
        self._current_path: str | None = None
        self.load(path)

    def _deserialize(self, path: str):
        with open(path, "rb") as f:
            engine = self._runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
            raise RuntimeError(
                f"TRT failed to deserialize {path} (free VRAM {free_gb:.1f} GB). "
                f"Probable cause: out of GPU memory."
            )
        ctx = engine.create_execution_context()
        if ctx is None:
            raise RuntimeError(f"TRT create_execution_context returned None for {path}")
        return engine, ctx

    def load(self, path):
        """Replace the active engine atomically. Releases the old engine first
        (necessary on tight VRAM), then loads the new one. If loading fails,
        attempts to restore the previously-active engine so the runtime keeps
        going instead of crashing on a NoneType context next frame."""
        previous_path = self._current_path

        self.context = None
        self.engine = None
        gc.collect()
        torch.cuda.empty_cache()

        try:
            self.engine, self.context = self._deserialize(path)
            self._current_path = path
        except Exception as load_err:
            self.engine = None
            self.context = None
            if previous_path and previous_path != path:
                try:
                    self.engine, self.context = self._deserialize(previous_path)
                    self._current_path = previous_path
                except Exception:
                    self.engine = None
                    self.context = None
                    self._current_path = None
            else:
                self._current_path = None
            raise load_err

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

        # Canonical SD 1.5 VAE for ALL models. The HyperVAE shipped with
        # RealisticVision uses a slightly different latent distribution than
        # vanilla SD 1.5; using it as the encoder for a DreamShaper or
        # MeinaMix UNet produces out-of-distribution latents and the UNet
        # outputs noise. sd-vae-ft-mse is the latent distribution every SD 1.5
        # finetune was trained against, so it works for every engine in the
        # model bank.
        print("Loading canonical SD 1.5 VAE (sd-vae-ft-mse)...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16,
        ).to("cuda")

        print("Loading TensorRT Engine...")
        self.unet_engine = StableTRT10Engine(
            engine_path, latent_h=self.latent_h, latent_w=self.latent_w, embed_dim=768,
        )

        self.shared_noise = torch.randn(
            (1, 4, self.latent_h, self.latent_w), device="cuda", dtype=torch.float16,
        )
        self.prev_output = None

    def swap_engine(self, engine_path: str) -> None:
        """Hot-swap the UNet engine without rebuilding the rest of the pipeline.
        Text encoder, VAE, and scheduler are kept (they're identical across
        SD 1.5 finetunes). Resets prev_output so a fresh style doesn't blend
        through the temporal smooth from the old model."""
        self.unet_engine.load(engine_path)
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
        # Safety net: if a previous engine swap left the engine unloaded, just
        # pass the warped input straight through this frame instead of
        # dereferencing a NoneType context.
        if self.unet_engine.context is None:
            return input_image
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
