import gc
import threading

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
    def __init__(
        self,
        model_path,
        engine_path,
        width=512,
        height=512,
        latent_scale=0.18215,
        loras: list[tuple[str, float]] | None = None,
    ):
        self.latent_scale = latent_scale
        self.latent_h = height // 8
        self.latent_w = width // 8

        print("Loading Stable Diffusion Pipeline...")
        self.pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            load_safety_checker=False,  # else the 1.2 GB checker is fetched
            use_safetensors=True,
        ).to("cuda")

        # Critical: fuse LCM LoRA (and any style LoRAs baked into the engine)
        # into the text encoder. At build time, every engine was compiled
        # against a text encoder with this exact LoRA stack fused. If we feed
        # them embeddings from an unfused (or differently-fused) text encoder,
        # the model's internal attention sees out-of-distribution conditioning
        # and the output devolves into noise. The UNet fuse is wasted (we
        # replace it with the TRT engine) but harmless.
        print("Fusing LCM + style LoRAs into text encoder...")
        self._fuse_loras_into_pipe(self.pipe, loras or [])
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

        # Per-engine text encoder cache. Each engine was compiled against its
        # own model's text encoder with a specific LoRA stack fused (LCM +
        # optional style LoRAs). Two engines that share a base checkpoint but
        # have different LoRAs still need DIFFERENT text encoders, so the key
        # includes the LoRA tuple — not just the checkpoint path.
        self._te_cache: dict[tuple, object] = {}
        # Held during swap_engine; step() try-acquires and bypasses if held.
        # Prevents step() from running with a half-swapped engine state.
        self._swap_lock = threading.Lock()

    @staticmethod
    def _fuse_loras_into_pipe(pipe, loras: list[tuple[str, float]]) -> None:
        """Fuse LCM LoRA followed by style LoRAs into pipe in place. Each fuse
        bakes the delta into the weights and is then unloaded so the next LoRA
        loads cleanly. Order and scales must match what builder.py used or the
        text encoder won't match the compiled engine."""
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        for lora_path, scale in loras:
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=scale)
            pipe.unload_lora_weights()

    def swap_engine(
        self,
        engine_path: str,
        checkpoint_path: str | None = None,
        loras: list[tuple[str, float]] | None = None,
    ) -> None:
        """Hot-swap the UNet engine. If checkpoint_path is provided, also swap
        the text encoder to one built from that checkpoint with the given
        LoRA stack fused, matching what the engine was compiled against.
        Order: load text encoder first (additive, safe to fail), then swap
        engine (releases old, may fail), then install text encoder. The whole
        operation is locked so step() can't run mid-swap."""
        loras = loras or []
        with self._swap_lock:
            # 1. Load new text encoder up-front. If this OOMs or otherwise
            # fails, we haven't touched the engine yet.
            new_te = None
            if checkpoint_path is not None:
                cache_key = (checkpoint_path, tuple(loras))
                if cache_key not in self._te_cache:
                    label = checkpoint_path + (
                        " + " + ", ".join(f"{p}@{s}" for p, s in loras) if loras else ""
                    )
                    print(f"Loading text encoder for {label}…")
                    tmp_pipe = StableDiffusionPipeline.from_single_file(
                        checkpoint_path,
                        torch_dtype=torch.float16,
                        safety_checker=None,
                        load_safety_checker=False,
                        use_safetensors=checkpoint_path.endswith(".safetensors"),
                    ).to("cuda")
                    self._fuse_loras_into_pipe(tmp_pipe, loras)
                    self._te_cache[cache_key] = tmp_pipe.text_encoder
                    del tmp_pipe
                    gc.collect()
                    torch.cuda.empty_cache()
                new_te = self._te_cache[cache_key]

            # 2. Swap engine (atomic with restore-on-failure inside load()).
            self.unet_engine.load(engine_path)

            # 3. Install matching text encoder.
            if new_te is not None:
                self.pipe.text_encoder = new_te

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

    def step(self, input_image, embeds, strength, delta,
             neg_embeds=None, neg_strength=0.0):
        """
        input_image: (1, 3, H, W) float16 CUDA tensor in [0, 1].
        Returns:     (1, 3, H, W) float16 CUDA tensor in [0, 1].

        neg_embeds + neg_strength implement embedding-space negative prompts:
        eff = pos + α·(pos − neg), pushing the conditioning AWAY from the
        negative direction in CLIP space. This is cheaper than true CFG
        (which would need a second UNet pass and halve FPS) and the only
        option that fits the batch=1 TRT engine without recompiling. Effect
        is weaker than CFG but it's free.
        """
        # If a swap is in flight, bypass UNet for this frame — the engine
        # and text encoder are momentarily inconsistent.
        if not self._swap_lock.acquire(blocking=False):
            return input_image
        try:
            if self.unet_engine.context is None:
                return input_image
            with torch.no_grad():
                x = input_image * 2.0 - 1.0
                init_latents = self.vae.encode(x).latent_dist.mode() * self.latent_scale

                # Fixed mid-range noise level — strength is handled in image space below.
                t = self.pipe.scheduler.timesteps[2]
                noised_latents = self.pipe.scheduler.add_noise(init_latents, self.shared_noise, t.unsqueeze(0))
                latent_model_input = self.pipe.scheduler.scale_model_input(noised_latents, t)

                if neg_embeds is not None and neg_strength > 0.0:
                    effective_embeds = embeds + neg_strength * (embeds - neg_embeds)
                else:
                    effective_embeds = embeds

                model_output = self.unet_engine(
                    latent_model_input.float(),
                    t.unsqueeze(0).float(),
                    effective_embeds.half(),
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
        finally:
            self._swap_lock.release()
