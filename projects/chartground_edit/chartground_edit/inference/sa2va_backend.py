"""Thin lazy-loading adapter for the Sa2VA InternVL3 Hugging Face export."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .mask_processing import MaskProcessingError, process_prediction_masks
from .prompt_variants import FULL_INSTRUCTION, PROMPT_TEMPLATES, build_prompt_variant
from .types import PredictionResult


MODEL_NAME = "ByteDance/Sa2VA-InternVL3-2B"
MIN_FREE_GPU_MEMORY_MB = 14_000.0
PROMPT_TEMPLATE = PROMPT_TEMPLATES[FULL_INSTRUCTION]


class Sa2VAError(RuntimeError):
    """Base exception for the ChartGround-Edit Sa2VA boundary."""


class Sa2VALoadError(Sa2VAError):
    """Raised when the local Sa2VA model or tokenizer cannot be loaded."""


def build_prompt(instruction: str) -> str:
    """Apply the one frozen Phase 2B-1 prompt without enriching instruction."""
    return build_prompt_variant(
        FULL_INSTRUCTION,
        instruction=instruction,
        referring_expression=instruction,
    )


class Sa2VAInternVL3Backend:
    """Lazy single-device backend around upstream ``predict_forward``.

    The implementation deliberately has no ground-truth input and does not use
    ``device_map``. Model and tokenizer are loaded once from the same local
    checkpoint, then retained for subsequent calls on this backend instance.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        model_name: str = MODEL_NAME,
        min_free_gpu_memory_mb: float = MIN_FREE_GPU_MEMORY_MB,
    ) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_dir():
            raise ValueError(f"checkpoint directory does not exist: {checkpoint}")
        if dtype != "bfloat16":
            raise ValueError("Sa2VAInternVL3Backend only supports dtype='bfloat16'")
        if not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("device must use logical CUDA syntax such as 'cuda:0'")
        self.checkpoint_path = checkpoint
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        self.min_free_gpu_memory_mb = float(min_free_gpu_memory_mb)
        if self.min_free_gpu_memory_mb < 0:
            raise ValueError("min_free_gpu_memory_mb must be non-negative")
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False
        self.model_load_attempts = 0
        self.model_load_time_ms: float | None = None
        self.pre_load_free_gpu_memory_mb: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load once from local files and move the complete model to one GPU."""
        if self._loaded:
            return
        self.model_load_attempts += 1
        started = time.perf_counter()
        try:
            self._prepare_cuda_for_load()
            self._perform_load()
            self._synchronize()
        except Exception as exc:
            self.model_load_time_ms = (time.perf_counter() - started) * 1000.0
            self._release_after_failure()
            raise Sa2VALoadError(
                f"failed to load {self.model_name} from local checkpoint: {exc}"
            ) from exc
        self.model_load_time_ms = (time.perf_counter() - started) * 1000.0
        self._loaded = True

    def predict_mask(
        self,
        image: Image.Image,
        instruction: str,
        parameters: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Call upstream ``predict_forward`` and normalize its first mask."""
        prompt = build_prompt(instruction)
        return self.predict_prompt(
            image,
            prompt,
            instruction=instruction,
            parameters=parameters,
        )

    def predict_prompt(
        self,
        image: Image.Image,
        prompt: str,
        *,
        instruction: str,
        parameters: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Run an exact pre-registered prompt without consulting ground truth."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a Pillow Image")
        if image.mode != "RGB":
            raise ValueError(f"image mode must be RGB, got {image.mode!r}")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if type(instruction) is not str or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if parameters not in (None, {}):
            raise ValueError("Sa2VAInternVL3Backend does not accept inference parameters yet")

        try:
            self.load()
        except Sa2VALoadError as exc:
            cause = exc.__cause__
            return self._failure_result(
                instruction,
                prompt,
                "model_load_failed",
                peak_gpu_memory_mb=self._peak_gpu_memory_mb(),
                metadata={
                    "stage": "model_load",
                    "error_type": type(cause).__name__ if cause is not None else type(exc).__name__,
                    "cuda_oom": self._is_cuda_oom(cause or exc),
                    "error": str(exc),
                },
            )

        torch = self._import_torch()
        started = time.perf_counter()
        raw_output: Any = None
        try:
            self._synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                raw_output = self._model.predict_forward(
                    image=image,
                    text=prompt,
                    tokenizer=self._tokenizer,
                )
            self._synchronize()
            inference_time_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            try:
                self._synchronize()
            except Exception:
                pass
            inference_time_ms = (time.perf_counter() - started) * 1000.0
            peak_gpu_memory_mb = self._peak_gpu_memory_mb()
            cuda_oom = self._is_cuda_oom(exc)
            if cuda_oom:
                self._release_after_failure()
            return self._failure_result(
                instruction,
                prompt,
                "model_inference_failed",
                inference_time_ms=inference_time_ms,
                peak_gpu_memory_mb=peak_gpu_memory_mb,
                metadata={
                    "stage": "predict_forward",
                    "error_type": type(exc).__name__,
                    "cuda_oom": cuda_oom,
                    "error": str(exc),
                },
            )

        if not isinstance(raw_output, dict):
            return self._failure_result(
                instruction,
                prompt,
                "invalid_model_output",
                inference_time_ms=inference_time_ms,
                peak_gpu_memory_mb=self._peak_gpu_memory_mb(),
                metadata={"output_type": type(raw_output).__name__},
            )

        text_output = raw_output.get("prediction")
        raw_masks = raw_output.get("prediction_masks")
        output_metadata: dict[str, Any] = {
            "prediction_type": type(text_output).__name__,
            "prediction_masks_present": "prediction_masks" in raw_output,
            "prediction_masks_type": type(raw_masks).__name__,
            "seg_token_count": text_output.count("[SEG]") if isinstance(text_output, str) else None,
            "pre_load_free_gpu_memory_mb": self.pre_load_free_gpu_memory_mb,
        }
        if "prediction_masks" not in raw_output:
            processed = process_prediction_masks(None, image_size=image.size)
        else:
            try:
                processed = process_prediction_masks(raw_masks, image_size=image.size)
            except MaskProcessingError as exc:
                return PredictionResult(
                    mask=None,
                    raw_masks=raw_masks,
                    text_output=text_output if isinstance(text_output, str) else None,
                    model_name=self.model_name,
                    checkpoint_path=str(self.checkpoint_path),
                    instruction=instruction,
                    prompt=prompt,
                    num_masks=len(raw_masks) if isinstance(raw_masks, (list, tuple)) else 0,
                    inference_time_ms=inference_time_ms,
                    model_load_time_ms=self.model_load_time_ms,
                    peak_gpu_memory_mb=self._peak_gpu_memory_mb(),
                    success=False,
                    failure_reason="invalid_prediction_mask",
                    raw_mask_shapes=exc.raw_mask_shapes,
                    metadata={
                        **output_metadata,
                        "raw_mask_metadata": exc.raw_mask_metadata,
                        "mask_processing_error": str(exc),
                    },
                )

        output_metadata.update(
            {
                "raw_mask_metadata": processed.raw_mask_metadata,
                "selected_mask_index": 0 if processed.num_masks else None,
                "mask_resized_nearest": processed.resized,
                "binarization": processed.binarization,
            }
        )
        failure_reason = processed.failure_reason
        if (
            failure_reason == "empty_prediction_masks"
            and isinstance(text_output, str)
            and "[SEG]" not in text_output
        ):
            output_metadata["mask_processing_failure"] = failure_reason
            failure_reason = "missing_seg_token"
        return PredictionResult(
            mask=processed.mask,
            raw_masks=raw_masks,
            text_output=text_output if isinstance(text_output, str) else None,
            model_name=self.model_name,
            checkpoint_path=str(self.checkpoint_path),
            instruction=instruction,
            prompt=prompt,
            num_masks=processed.num_masks,
            inference_time_ms=inference_time_ms,
            model_load_time_ms=self.model_load_time_ms,
            peak_gpu_memory_mb=self._peak_gpu_memory_mb(),
            success=failure_reason is None,
            failure_reason=failure_reason,
            raw_mask_shapes=processed.raw_mask_shapes,
            metadata=output_metadata,
        )

    def _perform_load(self) -> None:
        torch = self._import_torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        common = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path), **common)
        model = AutoModelForCausalLM.from_pretrained(
            str(self.checkpoint_path),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            **common,
        )
        model.eval()
        model.to(self.device)
        self._tokenizer = tokenizer
        self._model = model

    def _prepare_cuda_for_load(self) -> None:
        torch = self._import_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to the current process")
        device_index = int(self.device.split(":", 1)[1])
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"logical device {self.device} is unavailable; visible count is "
                f"{torch.cuda.device_count()}"
            )
        torch.cuda.set_device(device_index)
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
        self.pre_load_free_gpu_memory_mb = float(free_bytes / (1024.0**2))
        if self.pre_load_free_gpu_memory_mb < self.min_free_gpu_memory_mb:
            raise RuntimeError(
                f"GPU gate failed: {self.pre_load_free_gpu_memory_mb:.1f} MiB free, "
                f"requires at least {self.min_free_gpu_memory_mb:.1f} MiB"
            )
        torch.cuda.reset_peak_memory_stats(device_index)
        torch.cuda.synchronize(device_index)

    def _synchronize(self) -> None:
        torch = self._import_torch()
        torch.cuda.synchronize(self.device)

    def _peak_gpu_memory_mb(self) -> float | None:
        try:
            torch = self._import_torch()
            return float(torch.cuda.max_memory_allocated(self.device) / (1024.0**2))
        except Exception:
            return None

    @staticmethod
    def _import_torch() -> Any:
        import torch

        return torch

    def _release_after_failure(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()
        try:
            self._import_torch().cuda.empty_cache()
        except Exception:
            pass

    def _is_cuda_oom(self, error: BaseException) -> bool:
        try:
            torch = self._import_torch()
            return isinstance(error, torch.cuda.OutOfMemoryError)
        except Exception:
            return False

    def _failure_result(
        self,
        instruction: str,
        prompt: str,
        reason: str,
        *,
        inference_time_ms: float | None = None,
        peak_gpu_memory_mb: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PredictionResult:
        result_metadata = {
            "pre_load_free_gpu_memory_mb": self.pre_load_free_gpu_memory_mb,
            **(metadata or {}),
        }
        return PredictionResult(
            mask=None,
            raw_masks=None,
            text_output=None,
            model_name=self.model_name,
            checkpoint_path=str(self.checkpoint_path),
            instruction=instruction,
            prompt=prompt,
            num_masks=0,
            inference_time_ms=inference_time_ms,
            model_load_time_ms=self.model_load_time_ms,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            success=False,
            failure_reason=reason,
            metadata=result_metadata,
        )
