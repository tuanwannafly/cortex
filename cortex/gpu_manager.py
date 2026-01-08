"""
Hybrid GPU Manager for Cortex Linux.

This module supports:
- Per-app NVIDIA PRIME offload environment variables
- Real GPU mode switching via system backends (when available)

Note: Real switching typically requires sudo and may require logout/reboot,
depending on the backend and desktop session.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

PRIME_OFFLOAD_ENV: dict[str, str] = {
    "__NV_PRIME_RENDER_OFFLOAD": "1",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    "__VK_LAYER_NV_optimus": "NVIDIA_only",
}


class GPUSwitchBackend(str, Enum):
    PRIME_SELECT = "prime-select"  # Ubuntu/Debian (nvidia-prime)
    SYSTEM76_POWER = "system76-power"  # Pop!_OS / System76
    NONE = "none"


@dataclass(frozen=True)
class GPUSwitchPlan:
    backend: GPUSwitchBackend
    target_mode: str  # "integrated" | "hybrid" | "nvidia"
    commands: list[list[str]]
    requires_restart: bool
    notes: str


class GPUAppMode(str, Enum):
    """Per-app GPU preference."""

    NVIDIA = "nvidia"  # Offload to NVIDIA when possible
    INTEGRATED = "integrated"  # Force integrated / no PRIME offload env


def _cortex_config_dir() -> Path:
    """Return Cortex config directory.

    Supports overriding for tests/portable installs via:
        - CORTEX_CONFIG_DIR
    """
    override = os.environ.get("CORTEX_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cortex"


def _config_path() -> Path:
    return _cortex_config_dir() / "config.yaml"


def _load_config() -> dict:
    """Load ~/.cortex/config.yaml (best-effort).

    - If PyYAML is present, we use it.
    - Otherwise we fall back to JSON (JSON is valid YAML 1.2).
    """
    path = _config_path()
    if not path.exists():
        return {}

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    if not text.strip():
        return {}

    # Prefer YAML if available.
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        # Fallback to JSON
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def _save_config(data: dict) -> None:
    """Atomic write for ~/.cortex/config.yaml."""
    cfg_dir = _cortex_config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Prefer YAML if available.
    serialized: str
    try:
        import yaml  # type: ignore

        serialized = yaml.safe_dump(data, sort_keys=False)
    except Exception:
        serialized = json.dumps(data, indent=2, sort_keys=False)

    path = _config_path()
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(cfg_dir), encoding="utf-8") as tf:
        tf.write(serialized)
        tmp = Path(tf.name)
    os.replace(tmp, path)


def _get_gpu_apps_map(cfg: dict) -> dict[str, str]:
    gpu = cfg.get("gpu") if isinstance(cfg.get("gpu"), dict) else {}
    apps = gpu.get("apps") if isinstance(gpu.get("apps"), dict) else {}
    # Normalize to str->str
    out: dict[str, str] = {}
    for k, v in apps.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def set_app_gpu_preference(app: str, mode: str) -> None:
    """Persist a per-app GPU preference into ~/.cortex/config.yaml."""
    app_name = (app or "").strip()
    if not app_name:
        raise ValueError("app name is required")

    m = mode.lower().strip()
    if m not in {GPUAppMode.NVIDIA.value, GPUAppMode.INTEGRATED.value}:
        raise ValueError("mode must be 'nvidia' or 'integrated'")

    cfg = _load_config()
    gpu = cfg.get("gpu") if isinstance(cfg.get("gpu"), dict) else {}
    apps = gpu.get("apps") if isinstance(gpu.get("apps"), dict) else {}
    apps[app_name] = m
    gpu["apps"] = apps
    cfg["gpu"] = gpu
    _save_config(cfg)


def remove_app_gpu_preference(app: str) -> bool:
    """Remove a per-app GPU preference. Returns True if removed."""
    app_name = (app or "").strip()
    if not app_name:
        return False
    cfg = _load_config()
    apps = _get_gpu_apps_map(cfg)
    if app_name not in apps:
        return False
    apps.pop(app_name, None)
    gpu = cfg.get("gpu") if isinstance(cfg.get("gpu"), dict) else {}
    gpu["apps"] = apps
    cfg["gpu"] = gpu
    _save_config(cfg)
    return True


def get_app_gpu_preference(app: str) -> str | None:
    """Return stored preference for app, if any."""
    app_name = (app or "").strip()
    if not app_name:
        return None
    cfg = _load_config()
    apps = _get_gpu_apps_map(cfg)
    val = apps.get(app_name)
    if val in {GPUAppMode.NVIDIA.value, GPUAppMode.INTEGRATED.value}:
        return val
    return None


def list_app_gpu_preferences() -> dict[str, str]:
    """Return mapping of app -> mode from config."""
    cfg = _load_config()
    apps = _get_gpu_apps_map(cfg)
    # Filter to known values.
    return {
        k: v for k, v in apps.items() if v in {GPUAppMode.NVIDIA.value, GPUAppMode.INTEGRATED.value}
    }


def get_per_app_gpu_env(
    *, app: str | None = None, use_nvidia: bool | None = None
) -> dict[str, str]:
    """Return environment variables for per-application GPU assignment.

    You can either:
      - pass `use_nvidia=True/False` explicitly, OR
      - pass `app=<name>` and omit use_nvidia to read ~/.cortex/config.yaml.

    Args:
        app: App name (used for lookup only).
        use_nvidia: Force NVIDIA offload env vars.
    """
    if use_nvidia is None and app:
        pref = get_app_gpu_preference(app)
        if pref == GPUAppMode.NVIDIA.value:
            use_nvidia = True
        elif pref == GPUAppMode.INTEGRATED.value:
            use_nvidia = False

    if not use_nvidia:
        return {}

    # Return a copy so callers can't mutate the module constant.
    return dict(PRIME_OFFLOAD_ENV)


def detect_gpu_switch_backend() -> GPUSwitchBackend:
    """
    Detect which system backend is available for real GPU mode switching.

    Returns:
        GPUSwitchBackend: Detected backend or NONE if unsupported.
    """
    if shutil.which("prime-select"):
        return GPUSwitchBackend.PRIME_SELECT
    if shutil.which("system76-power"):
        return GPUSwitchBackend.SYSTEM76_POWER
    return GPUSwitchBackend.NONE


def plan_gpu_mode_switch(target_mode: str) -> GPUSwitchPlan | None:
    """
    Build a command plan to switch GPU mode using the available backend.

    Supported target_mode values:
        - "integrated"
        - "hybrid"
        - "nvidia"

    Returns:
        GPUSwitchPlan | None: Plan if a backend is available, else None.
    """
    target = target_mode.lower().strip()
    if target not in {"integrated", "hybrid", "nvidia"}:
        raise ValueError(f"Invalid target_mode: {target_mode}")

    backend = detect_gpu_switch_backend()

    if backend == GPUSwitchBackend.PRIME_SELECT:
        # prime-select: intel | nvidia | on-demand
        mapping = {
            "integrated": "intel",
            "hybrid": "on-demand",
            "nvidia": "nvidia",
        }
        cmd = ["sudo", "prime-select", mapping[target]]
        return GPUSwitchPlan(
            backend=backend,
            target_mode=target,
            commands=[cmd],
            requires_restart=True,
            notes="Uses nvidia-prime prime-select. Logout/reboot may be required.",
        )

    if backend == GPUSwitchBackend.SYSTEM76_POWER:
        # system76-power graphics: integrated | nvidia | hybrid
        cmd = ["sudo", "system76-power", "graphics", target]
        return GPUSwitchPlan(
            backend=backend,
            target_mode=target,
            commands=[cmd],
            requires_restart=True,
            notes="Uses system76-power graphics. Restart is typically required.",
        )

    return None


def run_command_with_env(cmd: list[str], extra_env: Mapping[str, str] | None = None) -> int:
    """
    Run a command with optional extra environment variables.

    Args:
        cmd: Command argv list.
        extra_env: Extra env vars to merge into current environment.

    Returns:
        int: Process return code.
    """
    env = os.environ.copy()
    if extra_env:
        env.update(dict(extra_env))

    completed = subprocess.run(cmd, env=env)
    return int(completed.returncode)


def apply_gpu_mode_switch(plan: GPUSwitchPlan, *, execute: bool = False) -> int:
    """Apply a switch plan.

    Args:
        plan: A plan created by plan_gpu_mode_switch().
        execute: If False, do nothing and return 0 (dry-run).

    Returns:
        int: 0 if successful (or dry-run), otherwise the failing command's exit code.
    """
    if not execute:
        return 0
    for c in plan.commands:
        rc = run_command_with_env(c)
        if rc != 0:
            return rc
    return 0


def get_gpu_profile(mode: str) -> dict[str, bool]:
    return {
        "use_integrated": mode == "Integrated",
        "use_nvidia": mode == "NVIDIA",
        "hybrid": mode == "Hybrid",
    }
