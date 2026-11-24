"""Build commands for GPU coordination on the SSH target."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from rcc.config import DockerSpec, Profile
from rcc.errors import RemoteError
from rcc.ssh import build_docker_invocation, check_ssh_connection, run_remote_capture

DEFAULT_LOCK = "/tmp/rcc-gpu{N}.lock"
DEFAULT_IDLE_CHECKS = 3
DEFAULT_IDLE_INTERVAL = 120.0
DEFAULT_IDLE_MEMORY_MIB = 256


def parse_indices(raw: str) -> list[int]:
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", raw):
        raise ValueError("--gpus expects comma-separated GPU indices, e.g. 0,1")
    indices = [int(part) for part in raw.split(",")]
    if len(set(indices)) != len(indices):
        raise ValueError("--gpus must list each GPU index once")
    return indices


def docker_devices(indices: list[int]) -> str:
    devices = "device=" + ",".join(map(str, indices))
    # Docker parses this option as CSV, so the quotes must be in the argv token.
    return f'"{devices}"' if len(indices) > 1 else devices


def _remote_argv(mode: str, options: dict) -> list[str]:
    source = Path(__file__).with_name("_remote_gpu.py").read_text(encoding="utf-8")
    return ["python3", "-c", source, mode, json.dumps(options)]


def run_argv(
    profile: Profile,
    argv: list[str],
    *,
    script: str | None,
    work_dir: str,
    docker: DockerSpec | None,
    tty: bool,
    session: str | None,
    indices: list[int],
    lock: str,
    idle_checks: int,
    idle_interval: float,
    idle_memory_mib: int,
) -> list[str]:
    if docker is not None:
        docker_argv = shlex.split(
            build_docker_invocation(
                profile.remote_dir, work_dir, docker, argv, script=script, tty=tty
            )
        )
        # Separate create from start so the supervisor has the container ID before work begins.
        command = ["docker", "create", *docker_argv[2:]]
    else:
        command = ["bash", "-c", f"set -euo pipefail; {script}"] if script is not None else argv
    return _remote_argv(
        "run",
        {
            "command": command,
            "image": docker.image if docker else None,
            "tty": tty,
            "session": session,
            "indices": indices,
            "lock": lock,
            "idle_checks": idle_checks,
            "idle_interval": idle_interval,
            "idle_memory_mib": idle_memory_mib,
        },
    )


def list_gpus(profile: Profile, lock: str) -> list[dict]:
    argv = _remote_argv("list", {"lock": lock, "deny_gpus": profile.deny_gpus})
    result = run_remote_capture(profile, argv, cwd="/")
    check_ssh_connection(profile, result)
    if result.returncode:
        raise RemoteError(
            result.stderr.strip() or f"GPU inventory failed on {profile.host}",
            exit_code=result.returncode,
        )
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise RemoteError(f"invalid GPU inventory received from {profile.host}") from exc
