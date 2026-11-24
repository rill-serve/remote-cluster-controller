from __future__ import annotations

from pathlib import Path

import typer

from rcc.config import resolve_profile
from rcc.context import merge_cli_overrides
from rcc.errors import RemoteError
from rcc.ssh import run_remote


def shell(
    profile: str | None = typer.Option(None, "--profile"),
    host: str | None = typer.Option(None, "--host"),
    remote_dir: str | None = typer.Option(None, "--remote-dir"),
) -> None:
    """Open an interactive remote shell in remote_dir."""
    overrides = merge_cli_overrides(profile=profile, host=host, remote_dir=remote_dir)
    resolved = resolve_profile(
        Path.cwd(),
        profile_name=overrides.profile,
        host_override=overrides.host,
        remote_dir_override=overrides.remote_dir,
    )
    code = run_remote(resolved, [], tty=True)
    if code != 0:
        raise RemoteError("remote shell exited non-zero", exit_code=code)
