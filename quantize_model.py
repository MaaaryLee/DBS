"""
Post-training quantization toolkit for the TD3 actor network.
Replaces the MATLAB-dependent flow with an offline pipeline that:
    1. Rebuilds the actor from the policy archive
    2. Exports an FP32 baseline
    3. Produces weight-only INT8 (dynamic) and static INT8 variants
"""

from __future__ import annotations

import copy
import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.quantization as tq
from torch.ao.quantization import quantize_dynamic

POLICY_ROOT = Path("models/policies")


@dataclass
class ActorArtifacts:
    """Metadata for an exported actor variant."""

    size_mb: float
    latency_ms: Dict[str, float]
    checkpoint: Path
    storage_format: str
    latency_ms_core: Optional[Dict[str, float]] = None


class TD3Actor(nn.Module):
    """
    Minimal TD3 actor (two hidden layers with ReLU, tanh head).
    Mirrors the Stable-Baselines3 actor.mu module.
    """

    def __init__(self, obs_dim: int, hidden_dims: Iterable[int], action_dim: int) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)
        layers = []
        in_dim = obs_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, action_dim))
        self.backbone = nn.Sequential(*layers)

        self.obs_dim = obs_dim
        self.hidden_dims = tuple(hidden_dims)
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(dtype=torch.float32)
        return torch.tanh(self.backbone(x))


class QuantizableTD3Actor(TD3Actor):
    """TD3 actor with quant/dequant stubs for static PTQ."""

    def __init__(self, obs_dim: int, hidden_dims: Iterable[int], action_dim: int) -> None:
        super().__init__(obs_dim=obs_dim, hidden_dims=hidden_dims, action_dim=action_dim)
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(dtype=torch.float32)
        x = self.quant(x)
        x = self.backbone(x)
        x = self.dequant(x)
        return torch.tanh(x)
def _load_policy_weights(zip_path: Path) -> Dict[str, torch.Tensor]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Trained checkpoint missing: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        if "policy.pth" not in archive.namelist():
            raise FileNotFoundError(f"`policy.pth` missing inside {zip_path}")
        with archive.open("policy.pth", "r") as buffer:
            state_dict: Dict[str, torch.Tensor] = torch.load(
                buffer, map_location="cpu", weights_only=False
            )
    return state_dict


def _build_actor_from_state(state_dict: Dict[str, torch.Tensor]) -> TD3Actor:
    obs_dim = state_dict["actor.mu.0.weight"].shape[1]
    hidden1 = state_dict["actor.mu.0.weight"].shape[0]
    hidden2 = state_dict["actor.mu.2.weight"].shape[0]
    action_dim = state_dict["actor.mu.4.weight"].shape[0]

    actor = TD3Actor(obs_dim=obs_dim, hidden_dims=[hidden1, hidden2], action_dim=action_dim)
    actor.eval()

    with torch.no_grad():
        actor.backbone[0].weight.copy_(state_dict["actor.mu.0.weight"])
        actor.backbone[0].bias.copy_(state_dict["actor.mu.0.bias"])
        actor.backbone[2].weight.copy_(state_dict["actor.mu.2.weight"])
        actor.backbone[2].bias.copy_(state_dict["actor.mu.2.bias"])
        actor.backbone[4].weight.copy_(state_dict["actor.mu.4.weight"])
        actor.backbone[4].bias.copy_(state_dict["actor.mu.4.bias"])

    return actor


def _load_calibration_tensor(states_path: Path, obs_dim: int) -> torch.Tensor:
    if states_path.exists():
        states = np.load(states_path)
        if states.ndim != 2 or states.shape[1] != obs_dim:
            raise ValueError(
                f"Calibration data shape mismatch: expected (*, {obs_dim}), got {states.shape}"
            )
        return torch.from_numpy(states).to(torch.float32)

    rng = np.random.default_rng(seed=0)
    synthetic = rng.normal(loc=0.0, scale=1.0, size=(1024, obs_dim)).astype(np.float32)
    return torch.from_numpy(synthetic)


def _benchmark_module(module: nn.Module, sample: torch.Tensor, runs: int = 200) -> Dict[str, float]:
    module.eval()
    timings = []
    with torch.inference_mode():
        for _ in range(10):
            module(sample)
        for _ in range(runs):
            start = time.perf_counter()
            module(sample)
            timings.append((time.perf_counter() - start) * 1000.0)

    data = np.array(timings, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "std": float(data.std(ddof=0)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
    }


def _benchmark_quantized_core(
    quant_actor: "QuantizableTD3Actor", sample: torch.Tensor, runs: int = 200
) -> Dict[str, float]:
    """
    Measure latency of the quantized backbone only, bypassing Quant/DeQuant stubs.
    Provides a closer estimate of deployment runtime once inputs arrive pre-quantized.
    """

    core = quant_actor.backbone
    first_layer = core[0]
    if not hasattr(first_layer, "scale") or not hasattr(first_layer, "zero_point"):
        raise ValueError("First quantized layer lacks scale/zero_point attributes.")

    if sample.dim() == 1:
        sample = sample.unsqueeze(0)

    qsample = torch.quantize_per_tensor(
        sample,
        scale=float(first_layer.scale),
        zero_point=int(first_layer.zero_point),
        dtype=torch.quint8,
    )

    timings = []
    core.eval()
    with torch.inference_mode():
        for _ in range(10):
            core(qsample)
        for _ in range(runs):
            start = time.perf_counter()
            core(qsample)
            timings.append((time.perf_counter() - start) * 1000.0)

    data = np.array(timings, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "std": float(data.std(ddof=0)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
    }


def _module_state_size_mb(module: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return len(buffer.getvalue()) / (1024 * 1024)


def _export_module(module: nn.Module, path: Path) -> Tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        scripted = torch.jit.script(module)
        scripted.save(str(path))
        return path, "torchscript"
    except Exception:
        torch.save(module, path)
        return path, "pickle"


def _quantize_dynamic(actor: TD3Actor) -> nn.Module:
    return quantize_dynamic(actor, {nn.Linear}, dtype=torch.qint8)


def _quantize_static(actor: TD3Actor, calibration: torch.Tensor) -> nn.Module:
    if torch.backends.quantized.engine != "fbgemm":
        torch.backends.quantized.engine = "fbgemm"

    quant_actor = QuantizableTD3Actor(
        obs_dim=actor.obs_dim, hidden_dims=actor.hidden_dims, action_dim=actor.action_dim
    )
    quant_actor.backbone.load_state_dict(actor.backbone.state_dict())
    quant_actor.eval()

    quant_actor.qconfig = tq.get_default_qconfig(torch.backends.quantized.engine)
    tq.prepare(quant_actor, inplace=True)

    with torch.inference_mode():
        for batch in torch.split(calibration, 64):
            quant_actor(batch)

    tq.convert(quant_actor, inplace=True)
    return quant_actor


def quantize_td3_actor(
    hidden_dims: Tuple[int, int] = (32, 32),
    model_timesteps: int = 2500,
    states_path: Path = Path("states_eval.npy"),
) -> Dict[str, ActorArtifacts]:
    checkpoint_path = Path(f"models/TD3_{hidden_dims[0]}_{hidden_dims[1]}/{model_timesteps}.zip")
    raw_state = _load_policy_weights(checkpoint_path)
    base_actor = _build_actor_from_state(raw_state)
    calibration = _load_calibration_tensor(states_path, base_actor.obs_dim)
    single_sample = calibration[:1]

    artifacts: Dict[str, ActorArtifacts] = {}

    fp32_path = POLICY_ROOT / f"actor_fp32_{hidden_dims[0]}_{hidden_dims[1]}.pt"
    fp32_path, fp32_format = _export_module(base_actor, fp32_path)
    artifacts["fp32"] = ActorArtifacts(
        size_mb=_module_state_size_mb(base_actor),
        latency_ms=_benchmark_module(base_actor, single_sample),
        checkpoint=fp32_path,
        storage_format=fp32_format,
    )

    dyn_actor = _quantize_dynamic(copy.deepcopy(base_actor))
    dyn_path = POLICY_ROOT / f"actor_int8_dynamic_{hidden_dims[0]}_{hidden_dims[1]}.pt"
    dyn_path, dyn_format = _export_module(dyn_actor, dyn_path)
    artifacts["dynamic_int8"] = ActorArtifacts(
        size_mb=_module_state_size_mb(dyn_actor),
        latency_ms=_benchmark_module(dyn_actor, single_sample),
        checkpoint=dyn_path,
        storage_format=dyn_format,
    )

    static_actor = _quantize_static(copy.deepcopy(base_actor), calibration)
    static_path = POLICY_ROOT / f"actor_int8_static_{hidden_dims[0]}_{hidden_dims[1]}.pt"
    static_path, static_format = _export_module(static_actor, static_path)
    artifacts["static_int8"] = ActorArtifacts(
        size_mb=_module_state_size_mb(static_actor),
        latency_ms=_benchmark_module(static_actor, single_sample),
        checkpoint=static_path,
        storage_format=static_format,
        latency_ms_core=_benchmark_quantized_core(static_actor, single_sample),
    )

    summary_path = POLICY_ROOT / f"quantization_summary_{hidden_dims[0]}_{hidden_dims[1]}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        name: {
            "size_mb": art.size_mb,
            "latency_ms": art.latency_ms,
            "checkpoint": str(art.checkpoint),
            "storage_format": art.storage_format,
            **({"latency_ms_core": art.latency_ms_core} if art.latency_ms_core else {}),
        }
        for name, art in artifacts.items()
    }

    summary_path.write_text(json.dumps(report, indent=2))
    return artifacts


if __name__ == "__main__":
    try:
        results = quantize_td3_actor()
    except Exception as exc:
        print("[X] Quantization failed:", exc)
        raise SystemExit(1) from exc

    print("=" * 70)
    print("TD3 Actor Quantization Summary")
    print("=" * 70)
    for label, artifact in results.items():
        print(f"{label.upper():>15}:")
        print(f"    checkpoint: {artifact.checkpoint}")
        print(f"    format: {artifact.storage_format}")
        print(f"    size (MB): {artifact.size_mb:.4f}")
        print(
            f"    latency (ms): mean={artifact.latency_ms['mean']:.4f}, "
            f"p50={artifact.latency_ms['p50']:.4f}, "
            f"p90={artifact.latency_ms['p90']:.4f}"
        )
        if artifact.latency_ms_core:
            core = artifact.latency_ms_core
            print(
                f"    latency_core (ms): mean={core['mean']:.4f}, "
                f"p50={core['p50']:.4f}, "
                f"p90={core['p90']:.4f}"
            )
    print("=" * 70)

