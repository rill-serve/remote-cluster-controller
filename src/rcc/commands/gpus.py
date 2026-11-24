from __future__ import annotations

import json
from pathlib import Path

import typer

from rcc import gpu
from rcc.config import resolve_profile, validate_gpu_lock
from rcc.context import merge_cli_overrides


def gpus(
    gpu_lock: str | None = typer.Option(
        None,
        "--gpu-lock",
        metavar="PATH",
        help="Shared remote lock filename with a {N} placeholder.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit GPU state and ownership as JSON."),
    profile: str | None = typer.Option(None, "--profile"),
    host: str | None = typer.Option(None, "--host"),
    remote_dir: str | None = typer.Option(None, "--remote-dir"),
) -> None:
    """Show the host's NVIDIA GPUs, compute processes and active GPU claims."""
    overrides = merge_cli_overrides(profile=profile, host=host, remote_dir=remote_dir)
    resolved = resolve_profile(
        Path.cwd(),
        profile_name=overrides.profile,
        host_override=overrides.host,
        remote_dir_override=overrides.remote_dir,
        require_remote_dir=False,
    )
    lock = gpu_lock if gpu_lock is not None else resolved.gpu_lock or gpu.DEFAULT_LOCK
    try:
        validate_gpu_lock(lock)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    rows = gpu.list_gpus(resolved, lock)
    if json_output:
        typer.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        typer.echo("No NVIDIA GPUs found.")
        return
    typer.echo(
        f"{'GPU':<5} {'MEM (MiB)':<10} {'STATE':<8} {'OWNER':<12} {'SESSION':<20} "
        f"{'CONTAINER':<12} {'IMAGE':<22} {'STARTED':<25} COMPUTE PIDS"
    )
    for row in rows:
        claim = row["claim"] or {}
        state = "locked" if row["locked"] else "busy" if row["compute_pids"] else "free"
        memory = row["memory_used_mib"]
        owner = claim.get("owner") or ("unknown" if row["locked"] else "-")
        typer.echo(
            f"{row['index']:<5} {str(memory) if memory is not None else '-':<10} {state:<8} "
            f"{owner:<12} {claim.get('session') or '-':<20} "
            f"{(claim.get('container') or '-')[:12]:<12} {claim.get('image') or '-':<22} "
            f"{claim.get('started_at') or '-':<25} "
            f"{','.join(map(str, row['compute_pids'])) or '-'}"
            + (" (reserved)" if row["denied"] else "")
        )
