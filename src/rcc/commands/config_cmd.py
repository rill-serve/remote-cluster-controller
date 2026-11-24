from __future__ import annotations

import json
from pathlib import Path

import typer

from rcc.config import find_rcc_dir, get_field, profile_to_dict, resolve_profile
from rcc.context import merge_cli_overrides
from rcc.errors import ConfigError


def config(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the resolved profile as JSON (stable, machine-readable).",
    ),
    get: str | None = typer.Option(
        None,
        "--get",
        metavar="KEY",
        help="Print a single field (e.g. host, remote_dir, env.NAME, tunnel.remote_port, docker.image).",
    ),
    profile: str | None = typer.Option(None, "--profile"),
    host: str | None = typer.Option(None, "--host"),
    remote_dir: str | None = typer.Option(None, "--remote-dir"),
) -> None:
    """Print the resolved profile after CLI overrides.

    By default prints ``key = value`` lines. ``--json`` emits a stable JSON
    object; ``--get KEY`` prints a single value (no quoting), so wrappers can
    stop parsing free text (issue #2 P2 / issue #3 #5).
    """
    rcc_dir = find_rcc_dir(Path.cwd())
    if rcc_dir is None:
        raise ConfigError("no .rcc/ found (run 'rcc init')")
    overrides = merge_cli_overrides(profile=profile, host=host, remote_dir=remote_dir)
    resolved = resolve_profile(
        rcc_dir.parent,
        profile_name=overrides.profile,
        host_override=overrides.host,
        remote_dir_override=overrides.remote_dir,
    )
    if get is not None:
        typer.echo(get_field(resolved, get))
        return
    if json_output:
        typer.echo(json.dumps(profile_to_dict(resolved), indent=2, sort_keys=True))
        return
    typer.echo(f"host = {resolved.host!r}")
    typer.echo(f"remote_dir = {resolved.remote_dir!r}")
    typer.echo(f"ssh_control_persist = {resolved.ssh_control_persist!r}")
    typer.echo(f"ssh_control_dir = {resolved.ssh_control_dir!r}")
    if resolved.proxy_jump:
        typer.echo(f"proxy_jump = {resolved.proxy_jump!r}")
    if resolved.identity_file:
        typer.echo(f"identity_file = {resolved.identity_file!r}")
    if resolved.env:
        typer.echo("env = {")
        for key in sorted(resolved.env):
            typer.echo(f"    {key} = {resolved.env[key]!r}")
        typer.echo("}")
    if resolved.keep_remote:
        typer.echo(f"keep_remote = {resolved.keep_remote!r}")
    if resolved.gpu_lock is not None:
        typer.echo(f"gpu_lock = {resolved.gpu_lock!r}")
    if resolved.deny_gpus:
        typer.echo(f"deny_gpus = {list(resolved.deny_gpus)!r}")
    if resolved.tunnel is not None:
        typer.echo(
            f"tunnel = {{ remote_port={resolved.tunnel.remote_port}, "
            f"local_port={resolved.tunnel.local_port}, "
            f"remote_host={resolved.tunnel.remote_host!r} }}"
        )
    if resolved.docker is not None:
        docker_parts = []
        if resolved.docker.image:
            docker_parts.append(f"image={resolved.docker.image!r}")
        if resolved.docker.container:
            docker_parts.append(f"container={resolved.docker.container!r}")
        docker_parts.append(f"workdir={resolved.docker.workdir!r}")
        docker_parts.append(f"rm={resolved.docker.rm!r}")
        if resolved.docker.gpus:
            docker_parts.append(f"gpus={resolved.docker.gpus!r}")
        if resolved.docker.shm_size:
            docker_parts.append(f"shm_size={resolved.docker.shm_size!r}")
        if resolved.docker.network:
            docker_parts.append(f"network={resolved.docker.network!r}")
        if resolved.docker.args:
            docker_parts.append(f"args={resolved.docker.args!r}")
        typer.echo(f"docker = {{ {', '.join(docker_parts)} }}")
        if resolved.docker.mounts:
            typer.echo("docker.mounts = [")
            for mount in resolved.docker.mounts:
                typer.echo(
                    f"    {{ source={mount.source!r}, target={mount.target!r}, "
                    f"mode={mount.mode!r} }}"
                )
            typer.echo("]")
        if resolved.docker.env:
            typer.echo("docker.env = {")
            for key in sorted(resolved.docker.env):
                typer.echo(f"    {key} = {resolved.docker.env[key]!r}")
            typer.echo("}")
