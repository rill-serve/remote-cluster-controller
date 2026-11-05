from __future__ import annotations

from pathlib import Path

import typer

from rcc import bg
from rcc.config import Profile, find_rcc_dir, load_config, resolve_profile
from rcc.context import merge_cli_overrides
from rcc.errors import ConfigError, RemoteError


def _rcc_dir() -> Path:
    rcc_dir = find_rcc_dir(Path.cwd())
    if rcc_dir is None:
        raise ConfigError("no .rcc/ found (run 'rcc init')")
    return rcc_dir


def _other_profiles(rcc_dir: Path, resolved: Profile):
    """Configured profiles pointing at a DIFFERENT host/dir than ``resolved``."""
    config = load_config(rcc_dir / "config.toml")
    for pname, p in config.profiles.items():
        if (p.host, p.remote_dir) != (resolved.host, resolved.remote_dir):
            yield pname, p


def _cross_profile_lookup(rcc_dir: Path, resolved: Profile, name: str) -> Profile:
    """bg runs live per-host: an explicit session name shouldn't require
    remembering which ``--profile`` launched it. If the run does not exist
    under the resolved profile, probe the others (cheap ssh checks; hosts may
    be down — that just means the run isn't there) and use the one that has it.
    """
    if bg.exists(resolved, name):
        return resolved
    for pname, p in _other_profiles(rcc_dir, resolved):
        try:
            if bg.exists(p, name):
                typer.echo(f"resolving {name!r} on profile {pname!r} ({p.host})", err=True)
                return p
        except RemoteError:
            continue
    return resolved  # not found anywhere: let the command's own error speak


def _resolve_target(rcc_dir: Path, resolved: Profile, name: str | None) -> tuple[Profile, str]:
    """Resolve (profile, session name) for a bg command.

    With an explicit name: search other profiles when the resolved one doesn't
    have it. Without: default to the sole running session under the resolved
    profile, and when THAT finds none, the sole running session across ALL
    configured profiles (a launch with ``--profile sgs`` followed by a bare
    ``rcc bg logs`` is the canonical trap).
    """
    if name is not None:
        return _cross_profile_lookup(rcc_dir, resolved, name), name
    try:
        return resolved, bg.resolve_name(resolved, name)
    except RemoteError:
        found: list[tuple[str, Profile, str]] = []
        for pname, p in _other_profiles(rcc_dir, resolved):
            try:
                for s in bg.list_sessions(p):
                    found.append((pname, p, s.name))
            except RemoteError:
                continue
        if len(found) == 1:
            pname, p, sname = found[0]
            typer.echo(
                f"using the sole running session {sname!r} on profile {pname!r} ({p.host})",
                err=True,
            )
            return p, sname
        raise


def _resolve(profile: str | None, host: str | None, remote_dir: str | None) -> Profile:
    rcc_dir = find_rcc_dir(Path.cwd())
    if rcc_dir is None:
        raise ConfigError("no .rcc/ found (run 'rcc init')")
    overrides = merge_cli_overrides(profile=profile, host=host, remote_dir=remote_dir)
    return resolve_profile(
        rcc_dir.parent,
        profile_name=overrides.profile,
        host_override=overrides.host,
        remote_dir_override=overrides.remote_dir,
    )


def ps(
    profile: str | None = typer.Option(None, "--profile"),
    host: str | None = typer.Option(None, "--host"),
    remote_dir: str | None = typer.Option(None, "--remote-dir"),
) -> None:
    """List running rcc detached sessions (tmux)."""
    rcc_dir = _rcc_dir()
    resolved = _resolve(profile, host, remote_dir)
    # Without overrides, aggregate every configured profile so sessions on
    # non-default hosts (e.g. sgs while default=beverin) are visible.
    if profile is None and host is None and remote_dir is None:
        targets = {name: p for name, p in load_config(rcc_dir / "config.toml").profiles.items()}
    else:
        targets = {"(resolved)": resolved}
    rows = []
    for pname, p in targets.items():
        try:
            rows.extend((pname, s) for s in bg.list_sessions(p))
        except RemoteError as exc:
            typer.echo(f"{pname}: unreachable ({exc})", err=True)
    if not rows:
        typer.echo("No running rcc detached sessions.")
        return
    typer.echo(f"{'NAME':<24} {'PROFILE':<12} {'CREATED':<20} ATTACHED")
    for pname, s in rows:
        typer.echo(f"{s.name:<24} {pname:<12} {s.created:<20} {s.attached}")


def logs(
    name: str | None = typer.Argument(None, help="Session name (default: the sole running one)."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow the log (tail -f)."),
    profile: str | None = typer.Option(None, "--profile"),
    host: str | None = typer.Option(None, "--host"),
    remote_dir: str | None = typer.Option(None, "--remote-dir"),
) -> None:
    """Tail the log of a detached run (in remote_dir/.rcc-runs/<name>.log)."""
    rcc_dir = _rcc_dir()
    resolved = _resolve(profile, host, remote_dir)
    resolved, target = _resolve_target(rcc_dir, resolved, name)
    code = bg.logs(resolved, target, follow=follow)
    if code != 0:
        raise RemoteError("tail exited non-zero", exit_code=code)


def attach(
    name: str | None = typer.Argument(None, help="Session name (default: the sole running one)."),
    profile: str | None = typer.Option(None, "--profile"),
    host: str | None = typer.Option(None, "--host"),
    remote_dir: str | None = typer.Option(None, "--remote-dir"),
) -> None:
    """Attach to a running detached session (tmux attach)."""
    rcc_dir = _rcc_dir()
    resolved = _resolve(profile, host, remote_dir)
    resolved, target = _resolve_target(rcc_dir, resolved, name)
    code = bg.attach(resolved, target)
    if code != 0:
        raise RemoteError("tmux attach exited non-zero", exit_code=code)


def wait(
    name: str | None = typer.Argument(None, help="Session name (default: the sole running one)."),
    timeout: float | None = typer.Option(
        None, "--timeout", metavar="SECONDS", help="Give up after SECONDS (exit 124)."
    ),
    profile: str | None = typer.Option(None, "--profile"),
    host: str | None = typer.Option(None, "--host"),
    remote_dir: str | None = typer.Option(None, "--remote-dir"),
) -> None:
    """Block until a detached run finishes; exit with its exit code.

    Non-zero on a failed/killed run; 124 on timeout. This closes the
    launch→monitor loop for non-SLURM hosts in one command (issue #3 #2).
    """
    resolved = _resolve(profile, host, remote_dir)
    rcc_dir = _rcc_dir()
    resolved, target = _resolve_target(rcc_dir, resolved, name)
    code = bg.wait(resolved, target, timeout=timeout)
    if code is None:
        typer.echo(f"timed out waiting for {target!r}", err=True)
        raise typer.Exit(code=124)
    if code != 0:
        raise RemoteError(f"detached run {target!r} exited with code {code}", exit_code=code)


def stop(
    name: str | None = typer.Argument(None, help="Session name (default: the sole running one)."),
    profile: str | None = typer.Option(None, "--profile"),
    host: str | None = typer.Option(None, "--host"),
    remote_dir: str | None = typer.Option(None, "--remote-dir"),
) -> None:
    """Kill a detached run (tmux kill-session)."""
    rcc_dir = _rcc_dir()
    resolved = _resolve(profile, host, remote_dir)
    resolved, target = _resolve_target(rcc_dir, resolved, name)
    code = bg.stop(resolved, target)
    if code != 0:
        raise RemoteError("tmux kill-session exited non-zero", exit_code=code)
    typer.echo(f"Stopped {target}.")
