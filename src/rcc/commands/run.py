from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import typer

from rcc import bg, gpu
from rcc.config import DockerMount, DockerSpec, resolve_profile, validate_gpu_lock
from rcc.context import merge_cli_overrides
from rcc.errors import RemoteError
from rcc.ssh import run_remote, run_remote_tee


def run(
    ctx: typer.Context,
    tty: bool = typer.Option(False, "-t", "--tty", help="Allocate a PTY on the remote"),
    shell: str | None = typer.Option(
        None,
        "-s",
        "--shell",
        help="Interpret the value as a shell snippet, e.g. -s 'squeue -u $USER | head'",
    ),
    env: list[str] | None = typer.Option(
        None,
        "--env",
        metavar="KEY=VAL",
        help="Export KEY=VAL on the remote before the command (repeatable). "
        "Merged on top of the profile env defaults.",
    ),
    env_file: Path | None = typer.Option(
        None,
        "--env-file",
        exists=True,
        readable=True,
        help="Load KEY=VAL lines from a file (like a .env; # comments allowed).",
    ),
    cwd: str | None = typer.Option(
        None,
        "--cwd",
        help="Working directory on the remote (default: profile remote_dir). "
        "Absolute path, or relative to remote_dir.",
    ),
    detach: bool = typer.Option(
        False,
        "--detach",
        help="Launch detached in a tmux session instead of running in the foreground "
        "(non-SLURM hosts). Manage with `rcc bg ps/logs/attach/wait/stop`. "
        "The run survives disconnects.",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help="Name for a detached run (default: auto-generated). Sanitized for tmux.",
    ),
    result_json: Path | None = typer.Option(
        None,
        "--result-json",
        metavar="PATH",
        help="Stream output live AND write {returncode, stdout, stderr, command} to PATH "
        "on exit — so wrappers get both live output and the captured text (tee).",
    ),
    docker: bool = typer.Option(
        False,
        "--docker",
        help="Run through the profile docker target, or through Docker CLI overrides.",
    ),
    docker_image: str | None = typer.Option(
        None,
        "--docker-image",
        help="Run in this image using Docker on the SSH remote host.",
    ),
    docker_container: str | None = typer.Option(
        None,
        "--docker-container",
        help="Execute in this existing Docker container on the SSH remote host.",
    ),
    docker_workdir: str | None = typer.Option(
        None,
        "--docker-workdir",
        help="Working directory inside the container (default: profile docker.workdir).",
    ),
    docker_mount: list[str] | None = typer.Option(
        None,
        "--docker-mount",
        metavar="SRC:DST[:ro|rw]",
        help="Add a docker run bind mount. SRC is a remote-host path; "
        "{remote_dir} and {cwd} are expanded remotely.",
    ),
    docker_env: list[str] | None = typer.Option(
        None,
        "--docker-env",
        metavar="KEY=VAL",
        help="Set container-only KEY=VAL (repeatable). Overrides docker env and --env.",
    ),
    docker_gpus: str | None = typer.Option(
        None,
        "--docker-gpus",
        help="Pass --gpus to docker run, e.g. all.",
    ),
    gpus: str | None = typer.Option(
        None, "--gpus", metavar="INDEX[,INDEX...]", help="Select and lock host GPUs by index."
    ),
    wait_idle_gpus: bool = typer.Option(
        False, "--wait-idle-gpus", help="Wait until the selected GPUs are idle and unlocked."
    ),
    idle_checks: int = typer.Option(
        gpu.DEFAULT_IDLE_CHECKS, "--idle-checks", min=1, help="Consecutive idle checks required."
    ),
    idle_interval: float = typer.Option(
        gpu.DEFAULT_IDLE_INTERVAL, "--idle-interval", min=0.01, help="Seconds between idle checks."
    ),
    idle_memory_mib: int = typer.Option(
        gpu.DEFAULT_IDLE_MEMORY_MIB,
        "--idle-memory-mib",
        min=0,
        help="Maximum memory in use per GPU for an idle check.",
    ),
    gpu_lock: str | None = typer.Option(
        None,
        "--gpu-lock",
        metavar="PATH",
        help="Shared remote lock filename with a {N} placeholder.",
    ),
    docker_shm_size: str | None = typer.Option(
        None,
        "--docker-shm-size",
        help="Pass --shm-size to docker run, e.g. 16g.",
    ),
    docker_network: str | None = typer.Option(
        None,
        "--docker-network",
        help="Pass --network to docker run, e.g. host.",
    ),
    profile: str | None = typer.Option(None, "--profile"),
    host: str | None = typer.Option(None, "--host"),
    remote_dir: str | None = typer.Option(None, "--remote-dir"),
) -> None:
    """Run a command on the remote, inside remote_dir.

    By default the arguments after `--` are passed as separate tokens. Use
    -s/--shell to pass a shell snippet that may contain pipelines, variable
    expansion, or quotes. ``--env KEY=VAL`` / ``--env-file`` inject environment
    on the remote (layered over the profile env defaults).
    ``--docker`` runs the same command through the resolved profile's
    Docker target; Docker CLI flags override the profile for one-off runs.

    ``--detach`` launches the command in a tmux session so it survives
    disconnects; manage it with ``rcc bg`` (issue #3 #2). ``--result-json PATH``
    tees output to the TTY while writing a structured result, so a wrapper gets
    live output *and* the captured text (issue #2 P1).
    """
    argv = list(ctx.args)
    if shell is None and not argv:
        raise typer.BadParameter(
            "missing remote command (use -- to separate, or -s for a shell snippet)"
        )
    if shell is not None and argv:
        raise typer.BadParameter("-s/--shell cannot be combined with extra positional arguments")
    overrides = merge_cli_overrides(profile=profile, host=host, remote_dir=remote_dir)
    resolved = resolve_profile(
        Path.cwd(),
        profile_name=overrides.profile,
        host_override=overrides.host,
        remote_dir_override=overrides.remote_dir,
    )
    cli_env = _resolve_env({}, env or [], env_file)
    merged_env = dict(resolved.env)
    merged_env.update(cli_env)
    docker_target = _resolve_docker_target(
        resolved.docker,
        enabled=docker,
        image=docker_image,
        container=docker_container,
        workdir=docker_workdir,
        mounts=docker_mount or [],
        inherited_env=resolved.env,
        cli_env=cli_env,
        docker_env_flags=docker_env or [],
        gpus=docker_gpus,
        shm_size=docker_shm_size,
        network=docker_network,
    )

    if detach and tty:
        raise typer.BadParameter("--tty is meaningless with --detach (the session has its own pty)")
    session_name = (name or _auto_name(shell, argv)) if detach else None
    if gpus is not None:
        if docker_gpus is not None:
            raise typer.BadParameter("--gpus and --docker-gpus are mutually exclusive")
        try:
            indices = gpu.parse_indices(gpus)
            lock = gpu_lock if gpu_lock is not None else resolved.gpu_lock or gpu.DEFAULT_LOCK
            validate_gpu_lock(lock)
            gpu_session = bg.sanitize(session_name) if session_name is not None else None
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if wait_idle_gpus and not math.isfinite(idle_interval):
            raise typer.BadParameter("--idle-interval must be finite")
        denied = sorted(set(indices).intersection(resolved.deny_gpus))
        if denied:
            raise typer.BadParameter(f"profile denies GPU indices: {', '.join(map(str, denied))}")
        if docker_target is not None:
            if docker_target.container:
                raise typer.BadParameter("--gpus requires a Docker image when using Docker")
            _validate_managed_docker_args(docker_target.args)
            docker_target = dataclasses.replace(docker_target, gpus=gpu.docker_devices(indices))
        remote_argv = gpu.run_argv(
            resolved,
            argv,
            script=shell,
            work_dir=bg.resolve_cwd(resolved.remote_dir, cwd)
            if detach
            else cwd or resolved.remote_dir,
            docker=docker_target,
            tty=tty and result_json is None,
            session=gpu_session,
            indices=indices,
            lock=lock,
            idle_checks=idle_checks if wait_idle_gpus else 0,
            idle_interval=idle_interval,
            idle_memory_mib=idle_memory_mib,
        )
        remote_script, docker_target = None, None
    else:
        if wait_idle_gpus or gpu_lock is not None:
            raise typer.BadParameter("--wait-idle-gpus and --gpu-lock require --gpus")
        if resolved.deny_gpus and (detach or docker_target is not None):
            raise typer.BadParameter("this profile reserves GPU indices; select GPUs with --gpus")
        if detach and docker_target is not None:
            raise typer.BadParameter("--docker is not supported with --detach yet")
        remote_argv, remote_script = argv, shell

    if session_name is not None:
        try:
            final = bg.start(
                resolved,
                session_name,
                remote_argv,
                script=remote_script,
                env=merged_env or None,
                cwd=cwd,
            )
        except ValueError as exc:  # bad name
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(
            f"Launched detached run {final!r} on {resolved.host}. "
            f"Logs: {bg.log_path(resolved, final)}. Try: rcc bg logs {final} -f"
        )
        return

    if result_json is not None:
        completed = run_remote_tee(
            resolved,
            remote_argv,
            script=remote_script,
            env=merged_env or None,
            cwd=cwd,
            docker=docker_target,
        )
        _write_result_json(result_json, completed, shell=shell, argv=argv)
        if completed.returncode != 0:
            raise RemoteError("remote command exited non-zero", exit_code=completed.returncode)
        return

    code = run_remote(
        resolved,
        remote_argv,
        script=remote_script,
        tty=tty,
        env=merged_env or None,
        cwd=cwd,
        docker=docker_target,
    )
    if code != 0:
        raise RemoteError("remote command exited non-zero", exit_code=code)


def _validate_managed_docker_args(args: list[str]) -> None:
    reserved = {
        "-d",
        "--detach",
        "--gpus",
        "--device",
        "--device-cgroup-rule",
        "--privileged",
        "--restart",
    }
    for arg in args:
        if arg.split("=", 1)[0] in reserved:
            raise typer.BadParameter(f"docker.args cannot set {arg!r} on GPU-managed runs")


def _auto_name(shell: str | None, argv: list[str]) -> str:
    """Generate a readable session name from the command (timestamp-suffixed)."""
    import time

    base = shell or (argv[0] if argv else "run")
    # Keep it tmux-safe and short.
    import re

    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", base.split()[0])[:24].strip("-") or "run"
    return f"{slug}-{int(time.time())}"


def _write_result_json(path: Path, completed, *, shell: str | None, argv: list[str]) -> None:
    payload = {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "command": shell if shell is not None else argv,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _resolve_env(
    profile_env: dict[str, str],
    env_flags: list[str],
    env_file: Path | None,
) -> dict[str, str]:
    merged: dict[str, str] = dict(profile_env)
    if env_file is not None:
        for key, value in _parse_env_file(env_file):
            merged[key] = value
    for item in env_flags:
        key, value = _split_env(item)
        merged[key] = value
    return merged


def _split_env(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise typer.BadParameter(f"--env expects KEY=VAL, got {item!r}")
    key, value = item.split("=", 1)
    if not key:
        raise typer.BadParameter(f"--env key is empty in {item!r}")
    return key, value


def _parse_env_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise typer.BadParameter(f"{path}:{lineno}: expected KEY=VAL")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"{path}:{lineno}: empty key")
        # Strip one layer of matching surrounding quotes, if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        pairs.append((key, value))
    return pairs


def _resolve_docker_target(
    profile_docker: DockerSpec | None,
    *,
    enabled: bool,
    image: str | None,
    container: str | None,
    workdir: str | None,
    mounts: list[str],
    inherited_env: dict[str, str],
    cli_env: dict[str, str],
    docker_env_flags: list[str],
    gpus: str | None,
    shm_size: str | None,
    network: str | None,
) -> DockerSpec | None:
    if image and container:
        raise typer.BadParameter("--docker-image and --docker-container are mutually exclusive")

    has_cli_override = any(
        [
            image,
            container,
            workdir,
            mounts,
            docker_env_flags,
            gpus,
            shm_size,
            network,
        ]
    )
    if not enabled and not has_cli_override:
        return None

    base = profile_docker or DockerSpec()
    parsed_mounts = [_parse_docker_mount_flag(item) for item in mounts]
    container_env = dict(inherited_env)
    container_env.update(base.env)
    container_env.update(cli_env)
    for item in docker_env_flags:
        key, value = _split_env(item)
        container_env[key] = value

    final_image = image if image is not None else base.image
    final_container = container if container is not None else base.container
    final_mounts = [*base.mounts, *parsed_mounts]
    final_gpus = gpus if gpus is not None else base.gpus
    final_shm_size = shm_size if shm_size is not None else base.shm_size
    final_network = network if network is not None else base.network

    if image is not None:
        final_container = None
    if container is not None:
        final_image = None
        if profile_docker is not None and profile_docker.image:
            final_mounts = parsed_mounts
            final_gpus = gpus
            final_shm_size = shm_size
            final_network = network

    if final_image and final_container:
        raise typer.BadParameter("docker image and container are mutually exclusive")
    if final_image is None and final_container is None:
        raise typer.BadParameter(
            "--docker requires [docker].image/[docker].container, "
            "--docker-image, or --docker-container"
        )

    return dataclasses.replace(
        base,
        image=final_image,
        container=final_container,
        workdir=workdir or base.workdir,
        mounts=final_mounts,
        env=container_env,
        gpus=final_gpus,
        shm_size=final_shm_size,
        network=final_network,
    )


def _parse_docker_mount_flag(raw: str) -> DockerMount:
    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        raise typer.BadParameter(f"--docker-mount expects SRC:DST[:ro|rw], got {raw!r}")
    source, target = parts[0], parts[1]
    if not source or not target:
        raise typer.BadParameter(f"--docker-mount expects non-empty SRC and DST, got {raw!r}")
    mode = parts[2] if len(parts) == 3 else "rw"
    if mode not in {"ro", "rw"}:
        raise typer.BadParameter(f"--docker-mount mode must be 'ro' or 'rw', got {mode!r}")
    return DockerMount(source=source, target=target, mode=mode)
