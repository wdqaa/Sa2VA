"""Small runtime contracts for the future Phase 4B one-step run."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from mmengine.hooks import Hook
from mmengine.optim import AmpOptimWrapper

from projects.sa2va.models.sa2va import Sa2VAModel


class ChartGroundStrategyAModel(Sa2VAModel):
    """Sa2VA strategy A: train and save only ``text_hidden_fcs``."""

    def __init__(
        self,
        *args: Any,
        expected_trainable_parameter_count: int,
        alignment_metadata: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.requires_grad_(False)
        self.text_hidden_fcs.requires_grad_(True)
        self.expected_trainable_parameter_count = expected_trainable_parameter_count
        self.alignment_metadata = dict(alignment_metadata)
        self.assert_strategy_a_trainables()

    def assert_strategy_a_trainables(self) -> tuple[list[str], int]:
        names = [name for name, param in self.named_parameters() if param.requires_grad]
        invalid = [name for name in names if not name.startswith("text_hidden_fcs.")]
        if invalid:
            raise RuntimeError(
                f"strategy A found trainable parameters outside text_hidden_fcs: {invalid}"
            )
        count = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        if count != self.expected_trainable_parameter_count:
            raise RuntimeError(
                "strategy A trainable parameter count mismatch: "
                f"expected={self.expected_trainable_parameter_count}, actual={count}"
            )
        return names, count

    def state_dict(self, *args: Any, **kwargs: Any) -> OrderedDict:
        """Keep the training checkpoint limited to the strategy-A projection."""
        prefix = kwargs.pop("prefix", "")
        return self.text_hidden_fcs.state_dict(
            *args, prefix=prefix + "text_hidden_fcs.", **kwargs
        )


class FailFastAmpOptimWrapper(AmpOptimWrapper):
    """Reject non-finite total loss before backward or optimizer update."""

    def update_params(self, loss, step_kwargs=None, zero_kwargs=None) -> None:
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"loss must be a torch.Tensor, got {type(loss).__name__}")
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError(f"non-finite loss before backward: {loss.detach()}")
        return super().update_params(loss, step_kwargs, zero_kwargs)


class Phase4BContractHook(Hook):
    """Log the frozen alignment contract and persist it in checkpoint metadata."""

    def before_run(self, runner) -> None:
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        names, count = model.assert_strategy_a_trainables()
        runner.logger.info(
            "Phase4B alignment metadata: %s", model.alignment_metadata
        )
        runner.logger.info(
            "Phase4B trainable parameters: count=%d names=%s", count, names
        )

    def after_train_iter(
        self, runner, batch_idx: int, data_batch=None, outputs=None
    ) -> None:
        if outputs:
            for name, value in outputs.items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise FloatingPointError(
                        f"non-finite training output {name}: {value.detach()}"
                    )
        if torch.cuda.is_available():
            peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
            runner.message_hub.update_scalar("train/peak_memory_mib", peak_mib)

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        names, count = model.assert_strategy_a_trainables()
        checkpoint.setdefault("meta", {})["chartground_phase4b"] = {
            **model.alignment_metadata,
            "trainable_parameter_names": names,
            "trainable_parameter_count": count,
            "checkpoint_state": "text_hidden_fcs_only",
        }

