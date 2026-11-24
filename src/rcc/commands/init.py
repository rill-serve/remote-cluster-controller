from __future__ import annotations

from pathlib import Path

import typer

from rcc.errors import ConfigError

_DEFAULT_CONFIG = """\
default = "default"

[profiles.default]
host = "myhost"                # change me (ssh alias or user@host)
remote_dir = "/path/on/remote" # change me
# ssh_control_persist = "30m"
# ssh_control_dir = "~/.ssh/controlmasters"

# Bastion hop / explicit key (issue #2 P3):
# proxy_jump = "bastion.example.com"
# identity_file = "~/.ssh/id_ed25519"

# Shared GPU coordination on the remote host:
# gpu_lock = "/tmp/rcc-gpu{N}.lock"
# deny_gpus = [2]

# Per-profile env defaults honored by `rcc run` (issue #2 P1):
# [profiles.default.env]
# TRITON_CACHE_DIR = "/scratch/cache"

# Remote-owned paths to protect from --delete/--mirror (issue #2 P1):
# keep_remote = ["logs/", "cache/", "*.safetensors", "last_service.env"]

# Default port-forward for `rcc tunnel` (issue #2 P3):
# [profiles.default.tunnel]
# remote_port = 8080
# local_port = 18080        # optional; defaults to remote_port
# remote_host = "localhost" # optional; set to a compute node to reach it

# Optional Docker target for `rcc run --docker`:
# [profiles.default.docker]
# image = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
# workdir = "/workspace"
# gpus = "all"
# shm_size = "16g"
# network = "host"
#
# [[profiles.default.docker.mounts]]
# source = "{remote_dir}"
# target = "/workspace"
# mode = "rw"
#
# [profiles.default.docker.env]
# HF_HOME = "/workspace/.cache/huggingface"
"""

_DEFAULT_IGNORE = """\
.git/
.venv/
venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
node_modules/
.next/
dist/
build/
"""


def init(
    force: bool = typer.Option(False, "--force", help="Overwrite existing .rcc/ contents"),
) -> None:
    """Scaffold .rcc/config.toml and .rcc/rccignore in the current directory."""
    rcc_dir = Path.cwd() / ".rcc"
    if rcc_dir.exists() and not force:
        raise ConfigError(f"{rcc_dir} already exists (use --force to overwrite)")
    rcc_dir.mkdir(exist_ok=True)
    (rcc_dir / "config.toml").write_text(_DEFAULT_CONFIG)
    (rcc_dir / "rccignore").write_text(_DEFAULT_IGNORE)
    typer.echo(f"Created {rcc_dir}/config.toml and {rcc_dir}/rccignore.")
    typer.echo("Next: edit config.toml to set host and remote_dir, then run 'rcc push --dry-run'.")
    typer.echo("Tip: add '.rcc/' to your .gitignore.")
