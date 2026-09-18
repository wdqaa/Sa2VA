"""Strict Strategy-B projection plus last-layer attention LoRA contracts."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
from peft import LoraConfig, get_peft_model

from projects.sa2va.models.sa2va import Sa2VAModel


PROJECTION_KEYS = (
    "text_hidden_fcs.0.weight",
    "text_hidden_fcs.0.bias",
    "text_hidden_fcs.2.weight",
    "text_hidden_fcs.2.bias",
)
LORA_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def exact_lora_targets(total_layers: int, last_layers: int = 8) -> tuple[str, ...]:
    if type(total_layers) is not int or type(last_layers) is not int:
        raise TypeError("layer counts must be integers")
    if total_layers <= 0 or last_layers <= 0 or last_layers > total_layers:
        raise ValueError("invalid last-layer LoRA selection")
    return tuple(
        f"model.layers.{layer}.self_attn.{projection}"
        for layer in range(total_layers - last_layers, total_layers)
        for projection in LORA_PROJECTIONS
    )


def canonical_trainable_name(name: str) -> str:
    prefix = "mllm.model.language_model."
    if name.startswith(prefix):
        return "language_model." + name.removeprefix(prefix)
    return name


def _selected_lora_parameters(language_model, targets: tuple[str, ...]):
    base_model = language_model.get_base_model()
    selected = []
    for target in targets:
        module = base_model.get_submodule(target)
        for branch in ("lora_A", "lora_B"):
            adapters = getattr(module, branch, None)
            if adapters is None or "default" not in adapters:
                raise RuntimeError(f"LoRA adapter is missing for {target}.{branch}")
            selected.append(adapters["default"].weight)
    if len({id(parameter) for parameter in selected}) != len(targets) * 2:
        raise RuntimeError("LoRA parameter selection contains duplicates")
    return selected


class ChartGroundStrategyBModel(Sa2VAModel):
    """Train only the projection and exact last-eight-layer attention LoRA."""

    def __init__(
        self,
        *args: Any,
        expected_projection_parameter_count: int,
        expected_lora_parameter_count: int,
        expected_total_trainable_parameter_count: int,
        expected_trainable_parameter_count: int | None = None,
        lora_target_modules: list[str] | tuple[str, ...],
        alignment_metadata: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if expected_trainable_parameter_count is not None:
            raise ValueError(
                "Strategy B must use expected_total_trainable_parameter_count"
            )
        super().__init__(*args, **kwargs)
        incompatible = getattr(self, "pretrained_load_incompatible_keys", None)
        if incompatible is None:
            raise RuntimeError("strategy B requires a full pretrained PTH")
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "full pretrained PTH does not exactly match the pre-LoRA model: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        # InternVLMLLM deliberately delays PEFT wrapping so the full Sa2VA PTH
        # is loaded into the unwrapped language model first. Its upstream
        # constructor also leaves ``use_llm_lora`` false; Strategy B invokes
        # the existing public hook explicitly after that exact load.
        if not hasattr(self.mllm.model.language_model, "get_base_model"):
            if self.mllm.llm_lora_config is None:
                raise RuntimeError("Strategy B requires an LLM LoRA config")
            self.mllm.manual_prepare_llm_for_lora()
            self.mllm.use_llm_lora = True
        self.expected_projection_parameter_count = int(
            expected_projection_parameter_count
        )
        self.expected_lora_parameter_count = int(expected_lora_parameter_count)
        self.expected_total_trainable_parameter_count = int(
            expected_total_trainable_parameter_count
        )
        self.lora_target_modules = tuple(lora_target_modules)
        self.alignment_metadata = dict(alignment_metadata)

        language_model = self.mllm.model.language_model
        total_layers = int(language_model.get_base_model().config.num_hidden_layers)
        expected_targets = exact_lora_targets(total_layers, 8)
        if self.lora_target_modules != expected_targets:
            raise RuntimeError(
                "Strategy B LoRA targets must be the exact last eight decoder "
                f"attention layers: expected={expected_targets}, "
                f"actual={self.lora_target_modules}"
            )
        configured = tuple(
            language_model.peft_config["default"].target_modules
        )
        if set(configured) != set(expected_targets):
            raise RuntimeError(
                f"PEFT target mismatch: configured={configured}, expected={expected_targets}"
            )

        self.requires_grad_(False)
        self.text_hidden_fcs.requires_grad_(True)
        for parameter in _selected_lora_parameters(language_model, expected_targets):
            parameter.requires_grad_(True)
        self.assert_strategy_b_trainables()

    def assert_strategy_b_trainables(self) -> dict[str, Any]:
        named = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        projection = {
            name: parameter
            for name, parameter in named.items()
            if name in PROJECTION_KEYS
        }
        language_model = self.mllm.model.language_model
        selected_ids = {
            id(parameter)
            for parameter in _selected_lora_parameters(
                language_model, self.lora_target_modules
            )
        }
        lora = {
            name: parameter
            for name, parameter in named.items()
            if id(parameter) in selected_ids
        }
        unexpected = sorted(set(named) - set(projection) - set(lora))
        if set(projection) != set(PROJECTION_KEYS) or unexpected:
            raise RuntimeError(
                "Strategy B trainable-name contract failed: "
                f"projection={sorted(projection)}, unexpected={unexpected}"
            )
        projection_count = sum(parameter.numel() for parameter in projection.values())
        lora_count = sum(parameter.numel() for parameter in lora.values())
        total_count = projection_count + lora_count
        expected = (
            self.expected_projection_parameter_count,
            self.expected_lora_parameter_count,
            self.expected_total_trainable_parameter_count,
        )
        actual = (projection_count, lora_count, total_count)
        if actual != expected or len(lora) != len(self.lora_target_modules) * 2:
            raise RuntimeError(
                "Strategy B trainable count mismatch: "
                f"expected={expected}, actual={actual}, lora_tensors={len(lora)}"
            )
        return {
            "names": sorted(named),
            "projection_names": sorted(projection),
            "lora_names": sorted(lora),
            "projection_count": projection_count,
            "lora_count": lora_count,
            "total_count": total_count,
            "model_total_count": sum(parameter.numel() for parameter in self.parameters()),
        }


def strategy_b_state(model: ChartGroundStrategyBModel) -> OrderedDict[str, torch.Tensor]:
    contract = model.assert_strategy_b_trainables()
    parameters = dict(model.named_parameters())
    state = OrderedDict(
        (canonical_trainable_name(name), parameters[name].detach().cpu().clone())
        for name in contract["names"]
    )
    if len(state) != 68:
        raise RuntimeError(f"Strategy B checkpoint requires 68 tensors, got {len(state)}")
    return state


def save_strategy_b_checkpoint(
    model: ChartGroundStrategyBModel,
    path: str | Path,
    metadata: Mapping[str, Any],
) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {output}")
    contract = model.assert_strategy_b_trainables()
    payload = {
        "meta": {
            "chartground_phase7b": {
                **dict(metadata),
                "checkpoint_state": "text_hidden_fcs_plus_llm_lora",
                "trainable_parameter_names": [
                    canonical_trainable_name(name) for name in contract["names"]
                ],
                "projection_parameter_count": contract["projection_count"],
                "lora_parameter_count": contract["lora_count"],
                "trainable_parameter_count": contract["total_count"],
                "model_total_parameter_count": contract["model_total_count"],
            }
        },
        "state_dict": strategy_b_state(model),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary file: {temporary}")
    torch.save(payload, temporary)
    temporary.replace(output)
    return output


def _load_payload(path: str | Path) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"meta", "state_dict"}:
        raise ValueError("Strategy B checkpoint must contain only meta and state_dict")
    metadata = checkpoint.get("meta", {}).get("chartground_phase7b")
    state = checkpoint.get("state_dict")
    if not isinstance(metadata, dict) or not isinstance(state, Mapping):
        raise ValueError("invalid Strategy B checkpoint payload")
    return dict(metadata), state


def _copy_exact_state(
    parameters: Mapping[str, torch.nn.Parameter],
    state: Mapping[str, torch.Tensor],
) -> None:
    if set(state) != set(parameters):
        raise ValueError(
            "Strategy B checkpoint key mismatch: "
            f"missing={sorted(set(parameters)-set(state))}, "
            f"unknown={sorted(set(state)-set(parameters))}"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"checkpoint value is not a tensor: {name}")
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"Strategy B shape mismatch for {name}: "
                    f"checkpoint={tuple(tensor.shape)}, model={tuple(parameter.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def load_strategy_b_checkpoint(
    model: ChartGroundStrategyBModel, path: str | Path
) -> dict[str, Any]:
    metadata, state = _load_payload(path)
    contract = model.assert_strategy_b_trainables()
    named = dict(model.named_parameters())
    parameters = {
        canonical_trainable_name(name): named[name] for name in contract["names"]
    }
    _copy_exact_state(parameters, state)
    if metadata.get("trainable_parameter_count") != contract["total_count"]:
        raise ValueError("Strategy B trainable parameter count mismatch")
    return metadata


def prepare_hf_strategy_b_model(
    model,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> tuple[str, ...]:
    total_layers = int(model.config.llm_config.num_hidden_layers)
    targets = exact_lora_targets(total_layers, 8)
    if hasattr(model.language_model, "peft_config"):
        raise RuntimeError("HF language model already contains a PEFT adapter")
    model.language_model = get_peft_model(
        model.language_model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(targets),
            modules_to_save=None,
        ),
    )
    model.requires_grad_(False)
    model.eval()
    return targets


def load_strategy_b_checkpoint_into_hf_model(
    model,
    path: str | Path,
    *,
    expected_identity: Mapping[str, str],
) -> dict[str, Any]:
    metadata, state = _load_payload(path)
    actual_identity = {
        "source_hf_revision": metadata.get("source_hf_revision"),
        "full_pth_sha256": metadata.get("full_pth", {}).get("sha256"),
        "base_repo_id": metadata.get("base_checkpoint", {}).get("repo_id"),
        "base_revision": metadata.get("base_checkpoint", {}).get("revision"),
        "manifest_sha256": metadata.get("manifest_sha256"),
    }
    if dict(expected_identity) != actual_identity:
        raise ValueError(
            f"Strategy B checkpoint identity mismatch: "
            f"checkpoint={actual_identity}, expected={dict(expected_identity)}"
        )
    parameters = dict(model.named_parameters())
    expected = {
        name: parameter
        for name, parameter in parameters.items()
        if name in PROJECTION_KEYS
        or (
            name.startswith("language_model.")
            and (".lora_A.default.weight" in name or ".lora_B.default.weight" in name)
        )
    }
    if len(expected) != 68:
        raise RuntimeError(f"HF Strategy B model requires 68 adapter tensors, got {len(expected)}")
    _copy_exact_state(expected, state)
    return metadata
