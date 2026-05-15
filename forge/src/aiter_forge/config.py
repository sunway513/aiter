"""Load and validate forge.yaml configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GPUConfig:
    host: str = "localhost"
    user: str = ""
    ssh_key: str = ""
    jump_host: str = ""
    aiter_root: str = ""


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    api_key_env: str = "ANTHROPIC_API_KEY"
    model: str = "claude-sonnet-4-20250514"


@dataclass
class ForgeConfig:
    gpu: GPUConfig = field(default_factory=GPUConfig)
    llm: LLMConfig | None = None

    @property
    def mode(self) -> str:
        """Auto-detect mode: auto if LLM key available, else manual."""
        if self.llm and os.environ.get(self.llm.api_key_env):
            return "auto"
        return "manual"

    @property
    def is_local(self) -> bool:
        return self.gpu.host in ("localhost", "127.0.0.1", "")


def load(
    config_path: str = "forge.yaml",
    cli_overrides: dict | None = None,
) -> ForgeConfig:
    """Load config from YAML file, merge env vars and CLI overrides.

    Resolution order (first non-empty wins):
    1. CLI overrides
    2. forge.yaml values
    3. Environment variables (AITER_ROOT, REMOTE_HOST, REMOTE_USER)
    4. Built-in defaults
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(path.read_text()) or {}

    # Build GPUConfig
    gpu_raw = raw.get("gpu", {}) or {}
    gpu = GPUConfig(
        host=gpu_raw.get("host", "localhost") or "localhost",
        user=gpu_raw.get("user", "") or "",
        ssh_key=gpu_raw.get("ssh_key", "") or "",
        jump_host=gpu_raw.get("jump_host", "") or "",
        aiter_root=gpu_raw.get("aiter_root", "") or "",
    )

    # Env var fallbacks
    if not gpu.aiter_root:
        gpu.aiter_root = os.environ.get("AITER_ROOT", "")
    if not gpu.host or gpu.host == "localhost":
        env_host = os.environ.get("REMOTE_HOST", "")
        if env_host:
            gpu.host = env_host
    if not gpu.user:
        gpu.user = os.environ.get("REMOTE_USER", "")

    # CLI overrides (flat keys: host, user, aiter_root, etc.)
    if cli_overrides:
        for key in ("host", "user", "ssh_key", "jump_host", "aiter_root"):
            if key in cli_overrides and cli_overrides[key]:
                setattr(gpu, key, cli_overrides[key])

    # Build LLMConfig (optional)
    llm = None
    llm_raw = raw.get("llm")
    if llm_raw:
        llm = LLMConfig(
            provider=llm_raw.get("provider", "anthropic"),
            api_key_env=llm_raw.get("api_key_env", "ANTHROPIC_API_KEY"),
            model=llm_raw.get("model", "claude-sonnet-4-20250514"),
        )

    # Model override from env
    model_env = os.environ.get("AITER_FORGE_MODEL")
    if model_env:
        if llm is None:
            llm = LLMConfig(model=model_env)
        else:
            llm.model = model_env

    config = ForgeConfig(gpu=gpu, llm=llm)

    # Validation
    if not config.gpu.aiter_root:
        raise ValueError(
            "gpu.aiter_root is required. Set it in forge.yaml or AITER_ROOT env var."
        )
    if not config.is_local and not config.gpu.user:
        raise ValueError(
            f"gpu.user is required when host is remote ({config.gpu.host}). "
            "Set it in forge.yaml or REMOTE_USER env var."
        )

    # Set AITER_ROOT env var so $AITER_ROOT in target.yaml commands works
    if "AITER_ROOT" not in os.environ:
        os.environ["AITER_ROOT"] = config.gpu.aiter_root

    return config
