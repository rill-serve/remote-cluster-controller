from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rcc.errors import ConfigError


@dataclass(frozen=True)
class TunnelSpec:
    """A port-forward a profile can carry (backs ``rcc tunnel``)."""

    remote_port: int
    local_port: int | None = None
    remote_host: str = "localhost"


@dataclass(frozen=True)
class DockerMount:
    """A bind mount to apply when a profile runs commands through Docker."""

    source: str
    target: str
    mode: str = "rw"


@dataclass(frozen=True)
class DockerSpec:
    """Profile defaults for running a command in Docker on the remote host."""

    image: str | None = None
    container: str | None = None
    workdir: str = "/workspace"
    mounts: list[DockerMount] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    gpus: str | None = None
    shm_size: str | None = None
    network: str | None = None
    args: list[str] = field(default_factory=list)
    rm: bool = True


@dataclass(frozen=True)
class Profile:
    host: str
    remote_dir: str
    ssh_control_persist: str = "30m"
    ssh_control_dir: str = "~/.ssh/controlmasters"
    # Bastion hops are the norm on HPC; encapsulate them instead of relying on
    # an SSH-config dance the user has to remember (issue #2 P3).
    proxy_jump: str | None = None
    identity_file: str | None = None
    # Per-profile env defaults so serve scripts stop re-deriving constants
    # (TRITON_CACHE_DIR, redirected $HOME, ...). Honored by ``rcc run``.
    env: dict[str, str] = field(default_factory=dict)
    # Protect-list of remote-owned paths (logs/, cache/, *.safetensors, ...) that
    # survive even ``--mirror``. This is rccignore-with-teeth: the guard rail
    # lives in the tool, not the user's memory (issue #2 P1).
    keep_remote: list[str] = field(default_factory=list)
    # Default port-forward for ``rcc tunnel`` (issue #2 P3).
    tunnel: TunnelSpec | None = None
    # Optional Docker execution target for ``rcc run --docker``.
    docker: DockerSpec | None = None
    # Shared lock path template and GPU indices reserved on this host.
    gpu_lock: str | None = None
    deny_gpus: tuple[int, ...] = ()


@dataclass(frozen=True)
class Config:
    default: str
    profiles: dict[str, Profile] = field(default_factory=dict)


_REQUIRED = ("host", "remote_dir")


def load_config(path: Path) -> Config:
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: malformed TOML: {exc}") from exc
    except FileNotFoundError as exc:
        raise ConfigError(f"{path}: not found") from exc

    default = raw.get("default")
    if not isinstance(default, str) or not default:
        raise ConfigError(f"{path}: missing 'default' profile name")

    profiles_raw = raw.get("profiles", {})
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ConfigError(f"{path}: no profiles defined")

    profiles: dict[str, Profile] = {}
    for name, body in profiles_raw.items():
        if not isinstance(name, str) or not isinstance(body, dict):
            raise ConfigError(f"{path}: malformed profile entry")
        profiles[name] = _parse_profile(path, name, body)

    if default not in profiles:
        raise ConfigError(f"{path}: default profile '{default}' not defined")

    return Config(default=default, profiles=profiles)


def _parse_profile(path: Path, name: str, body: dict[str, Any]) -> Profile:
    for required in _REQUIRED:
        if required not in body:
            raise ConfigError(f"{path}: profile '{name}' missing required field '{required}'")
        if not isinstance(body[required], str) or not body[required]:
            raise ConfigError(f"{path}: profile '{name}' field '{required}' must be a string")

    optional: dict[str, str] = {}
    for key in ("ssh_control_persist", "ssh_control_dir", "proxy_jump", "identity_file"):
        value = body.get(key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ConfigError(f"{path}: profile '{name}' field '{key}' must be a string")
            optional[key] = value

    env = _parse_env(path, name, body.get("env"))
    keep_remote = _parse_keep_remote(path, name, body.get("keep_remote"))
    tunnel = _parse_tunnel(path, name, body.get("tunnel"))
    docker = _parse_docker(path, name, body.get("docker"))
    gpu_lock = body.get("gpu_lock")
    if gpu_lock is not None:
        try:
            validate_gpu_lock(gpu_lock)
        except ValueError as exc:
            raise ConfigError(f"{path}: profile '{name}' {exc}") from exc
    deny_gpus = body.get("deny_gpus", [])
    if not isinstance(deny_gpus, list) or any(
        type(index) is not int or index < 0 for index in deny_gpus
    ):
        raise ConfigError(f"{path}: profile '{name}' deny_gpus must be an array of GPU indices")

    return Profile(
        host=body["host"],
        remote_dir=body["remote_dir"],
        env=env,
        keep_remote=keep_remote,
        tunnel=tunnel,
        docker=docker,
        gpu_lock=gpu_lock,
        deny_gpus=tuple(deny_gpus),
        **optional,
    )


def validate_gpu_lock(template: str) -> None:
    if (
        not isinstance(template, str)
        or not template.startswith("/")
        or "{N}" not in template.rsplit("/", 1)[-1]
        or "\x00" in template
    ):
        raise ValueError("GPU lock path must be absolute and contain {N} in its filename")


def _parse_env(path: Path, name: str, raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: profile '{name}' field 'env' must be a table")
    env: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ConfigError(f"{path}: profile '{name}' env entry {key!r} must be a string")
        # Env values are strings by definition; accept scalar numerics so
        # ``PORT = 8080`` reads naturally. Reject bools/arrays/tables to avoid
        # surprises (TOML ``true`` would otherwise become "True").
        if isinstance(value, bool) or isinstance(value, (list, dict)):
            raise ConfigError(f"{path}: profile '{name}' env entry {key!r} must be a string")
        env[key] = value if isinstance(value, str) else str(value)
    return env


def _parse_keep_remote(path: Path, name: str, raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ConfigError(
            f"{path}: profile '{name}' field 'keep_remote' must be an array of glob strings"
        )
    return list(raw)


def _parse_tunnel(path: Path, name: str, raw: Any) -> TunnelSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: profile '{name}' field 'tunnel' must be a table")
    remote_port = raw.get("remote_port")
    if not isinstance(remote_port, int):
        raise ConfigError(f"{path}: profile '{name}' tunnel.remote_port must be an integer port")
    local_port = raw.get("local_port")
    if local_port is not None and not isinstance(local_port, int):
        raise ConfigError(f"{path}: profile '{name}' tunnel.local_port must be an integer port")
    remote_host = raw.get("remote_host")
    if remote_host is not None and (not isinstance(remote_host, str) or not remote_host):
        raise ConfigError(f"{path}: profile '{name}' tunnel.remote_host must be a string")
    return TunnelSpec(
        remote_port=remote_port,
        local_port=local_port,
        remote_host=remote_host if isinstance(remote_host, str) else "localhost",
    )


def _parse_docker(path: Path, name: str, raw: Any) -> DockerSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: profile '{name}' field 'docker' must be a table")

    image = _optional_string(path, name, raw, "docker.image")
    container = _optional_string(path, name, raw, "docker.container")
    if image and container:
        raise ConfigError(
            f"{path}: profile '{name}' docker.image and docker.container are mutually exclusive"
        )

    workdir = raw.get("workdir", "/workspace")
    if not isinstance(workdir, str) or not workdir:
        raise ConfigError(f"{path}: profile '{name}' docker.workdir must be a string")

    args = raw.get("args", [])
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ConfigError(f"{path}: profile '{name}' docker.args must be an array of strings")

    rm = raw.get("rm", True)
    if not isinstance(rm, bool):
        raise ConfigError(f"{path}: profile '{name}' docker.rm must be a boolean")

    return DockerSpec(
        image=image,
        container=container,
        workdir=workdir,
        mounts=_parse_docker_mounts(path, name, raw.get("mounts")),
        env=_parse_docker_env(path, name, raw.get("env")),
        gpus=_optional_string(path, name, raw, "docker.gpus"),
        shm_size=_optional_string(path, name, raw, "docker.shm_size"),
        network=_optional_string(path, name, raw, "docker.network"),
        args=list(args),
        rm=rm,
    )


def _optional_string(path: Path, name: str, raw: dict[str, Any], key: str) -> str | None:
    field = key.split(".")[-1]
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path}: profile '{name}' {key} must be a string")
    return value


def _parse_docker_mounts(path: Path, name: str, raw: Any) -> list[DockerMount]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigError(f"{path}: profile '{name}' docker.mounts must be an array")
    mounts: list[DockerMount] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"{path}: profile '{name}' docker.mounts[{idx}] must be a table")
        source = item.get("source")
        target = item.get("target")
        if not isinstance(source, str) or not source:
            raise ConfigError(
                f"{path}: profile '{name}' docker.mounts[{idx}].source must be a string"
            )
        if not isinstance(target, str) or not target:
            raise ConfigError(
                f"{path}: profile '{name}' docker.mounts[{idx}].target must be a string"
            )
        mode = item.get("mode", "rw")
        if mode not in {"ro", "rw"}:
            raise ConfigError(
                f"{path}: profile '{name}' docker.mounts[{idx}].mode must be 'ro' or 'rw'"
            )
        mounts.append(DockerMount(source=source, target=target, mode=mode))
    return mounts


def _parse_docker_env(path: Path, name: str, raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: profile '{name}' docker.env must be a table")
    env: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ConfigError(f"{path}: profile '{name}' docker.env entry {key!r} must be a string")
        if isinstance(value, bool) or isinstance(value, (list, dict)):
            raise ConfigError(f"{path}: profile '{name}' docker.env entry {key!r} must be a string")
        env[key] = value if isinstance(value, str) else str(value)
    return env


def find_rcc_dir(start: Path) -> Path | None:
    """Walk up from start until a .rcc/ directory is found."""
    current = start.resolve()
    if current.is_file():
        current = current.parent
    while True:
        candidate = current / ".rcc"
        if candidate.is_dir():
            return candidate
        if current.parent == current:
            return None
        current = current.parent


def resolve_profile(
    start: Path,
    *,
    profile_name: str | None = None,
    host_override: str | None = None,
    remote_dir_override: str | None = None,
) -> Profile:
    rcc_dir = find_rcc_dir(start)
    if rcc_dir is None:
        raise ConfigError("no .rcc/ found (run 'rcc init')")

    cfg = load_config(rcc_dir / "config.toml")
    name = profile_name or cfg.default
    if name not in cfg.profiles:
        raise ConfigError(f"profile '{name}' not defined in {rcc_dir}/config.toml")

    profile = cfg.profiles[name]
    overrides: dict[str, str] = {}
    if host_override:
        overrides["host"] = host_override
    if remote_dir_override:
        overrides["remote_dir"] = remote_dir_override
    return dataclasses.replace(profile, **overrides) if overrides else profile


def profile_to_dict(profile: Profile) -> dict[str, Any]:
    """Stable, JSON-serializable view of a resolved profile (for ``--json``)."""
    return dataclasses.asdict(profile)


def get_field(profile: Profile, key: str) -> str:
    """Fetch a single profile field by dotted key for ``config --get``.

    Supports top-level fields (``host``, ``remote_dir``, ``proxy_jump``,
    ``identity_file``, ``ssh_control_persist``, ``ssh_control_dir``) and the
    ``env.<NAME>`` / ``tunnel.<field>`` dotted forms so wrappers can read a
    single value without parsing free text.
    """
    if key.startswith("env."):
        env_key = key[len("env.") :]
        if env_key not in profile.env:
            raise ConfigError(f"env has no key {env_key!r}")
        return profile.env[env_key]
    if key.startswith("tunnel."):
        if profile.tunnel is None:
            raise ConfigError("profile defines no tunnel")
        sub = key[len("tunnel.") :]
        try:
            value = getattr(profile.tunnel, sub)
        except AttributeError as exc:
            raise ConfigError(f"tunnel has no field {sub!r}") from exc
        return "" if value is None else str(value)
    if key.startswith("docker."):
        if profile.docker is None:
            raise ConfigError("profile defines no docker")
        sub = key[len("docker.") :]
        if sub.startswith("env."):
            env_key = sub[len("env.") :]
            if env_key not in profile.docker.env:
                raise ConfigError(f"docker.env has no key {env_key!r}")
            return profile.docker.env[env_key]
        try:
            value = getattr(profile.docker, sub)
        except AttributeError as exc:
            raise ConfigError(f"docker has no field {sub!r}") from exc
        if isinstance(value, (dict, list)):
            raise ConfigError(f"docker.{sub} is structured; use --json")
        return "" if value is None else str(value)
    try:
        value = getattr(profile, key)
    except AttributeError as exc:
        raise ConfigError(f"unknown profile field {key!r}") from exc
    return "" if value is None else str(value)
