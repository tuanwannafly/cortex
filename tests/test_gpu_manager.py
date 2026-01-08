import os
from types import SimpleNamespace

import pytest


def test_detect_gpu_switch_backend_prime_select(monkeypatch):
    import cortex.gpu_manager as gm

    monkeypatch.setattr(
        gm.shutil,
        "which",
        lambda name: "/usr/bin/prime-select" if name == "prime-select" else None,
    )
    assert gm.detect_gpu_switch_backend() == gm.GPUSwitchBackend.PRIME_SELECT


def test_detect_gpu_switch_backend_system76_power(monkeypatch):
    import cortex.gpu_manager as gm

    def fake_which(name: str):
        if name == "prime-select":
            return None
        if name == "system76-power":
            return "/usr/bin/system76-power"
        return None

    monkeypatch.setattr(gm.shutil, "which", fake_which)
    assert gm.detect_gpu_switch_backend() == gm.GPUSwitchBackend.SYSTEM76_POWER


def test_plan_gpu_mode_switch_prime_select(monkeypatch):
    import cortex.gpu_manager as gm

    monkeypatch.setattr(gm, "detect_gpu_switch_backend", lambda: gm.GPUSwitchBackend.PRIME_SELECT)

    plan = gm.plan_gpu_mode_switch("hybrid")
    assert plan is not None
    assert plan.backend == gm.GPUSwitchBackend.PRIME_SELECT
    assert plan.target_mode == "hybrid"
    assert plan.commands == [["sudo", "prime-select", "on-demand"]]


def test_plan_gpu_mode_switch_system76_power(monkeypatch):
    import cortex.gpu_manager as gm

    monkeypatch.setattr(gm, "detect_gpu_switch_backend", lambda: gm.GPUSwitchBackend.SYSTEM76_POWER)

    plan = gm.plan_gpu_mode_switch("nvidia")
    assert plan is not None
    assert plan.backend == gm.GPUSwitchBackend.SYSTEM76_POWER
    assert plan.commands == [["sudo", "system76-power", "graphics", "nvidia"]]


def test_plan_gpu_mode_switch_invalid_mode_raises(monkeypatch):
    import cortex.gpu_manager as gm

    monkeypatch.setattr(gm, "detect_gpu_switch_backend", lambda: gm.GPUSwitchBackend.NONE)
    with pytest.raises(ValueError):
        gm.plan_gpu_mode_switch("banana")


def test_run_command_with_env_merges_env(monkeypatch):
    import cortex.gpu_manager as gm

    seen = {}

    def fake_run(cmd, env=None, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = dict(env or {})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gm.subprocess, "run", fake_run)

    rc = gm.run_command_with_env(["echo", "hi"], extra_env={"A": "1"})
    assert rc == 0
    assert seen["cmd"] == ["echo", "hi"]
    assert seen["env"].get("A") == "1"


def test_per_app_config_roundtrip(tmp_path, monkeypatch):
    import cortex.gpu_manager as gm

    monkeypatch.setenv("CORTEX_CONFIG_DIR", str(tmp_path))

    gm.set_app_gpu_preference("blender", "nvidia")
    gm.set_app_gpu_preference("firefox", "integrated")

    assert gm.get_app_gpu_preference("blender") == "nvidia"
    assert gm.get_app_gpu_preference("firefox") == "integrated"

    prefs = gm.list_app_gpu_preferences()
    assert prefs["blender"] == "nvidia"
    assert prefs["firefox"] == "integrated"

    env_blender = gm.get_per_app_gpu_env(app="blender")
    assert env_blender.get("__NV_PRIME_RENDER_OFFLOAD") == "1"

    env_firefox = gm.get_per_app_gpu_env(app="firefox")
    assert env_firefox == {}

    assert gm.remove_app_gpu_preference("blender") is True
    assert gm.get_app_gpu_preference("blender") is None


def test_set_app_gpu_preference_validates_input(tmp_path, monkeypatch):
    import cortex.gpu_manager as gm

    monkeypatch.setenv("CORTEX_CONFIG_DIR", str(tmp_path))

    with pytest.raises(ValueError):
        gm.set_app_gpu_preference("", "nvidia")

    with pytest.raises(ValueError):
        gm.set_app_gpu_preference("blender", "bad")
