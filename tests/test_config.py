from pathlib import Path

import pytest

from rcc.config import (
    DockerSpec,
    Profile,
    TunnelSpec,
    find_rcc_dir,
    get_field,
    load_config,
    profile_to_dict,
    resolve_profile,
)
from rcc.errors import ConfigError


def test_load_config_parses_profiles(rcc_project: Path):
    cfg = load_config(rcc_project / ".rcc" / "config.toml")
    assert cfg.default == "alpha"
    assert cfg.profiles["alpha"].host == "alpha.example.com"
    assert cfg.profiles["alpha"].remote_dir == "/srv/alpha"


def test_load_config_applies_field_defaults(rcc_project: Path):
    cfg = load_config(rcc_project / ".rcc" / "config.toml")
    profile = cfg.profiles["alpha"]
    assert profile.ssh_control_persist == "30m"
    assert profile.ssh_control_dir == "~/.ssh/controlmasters"


def test_load_config_rejects_missing_required_field(tmp_path: Path):
    config = tmp_path / "c.toml"
    config.write_text('default = "a"\n[profiles.a]\nhost = "h"\n')
    with pytest.raises(ConfigError, match="remote_dir"):
        load_config(config)


def test_load_config_rejects_unknown_default(tmp_path: Path):
    config = tmp_path / "c.toml"
    config.write_text('default = "zzz"\n[profiles.a]\nhost = "h"\nremote_dir = "/d"\n')
    with pytest.raises(ConfigError, match="zzz"):
        load_config(config)


def test_load_config_rejects_empty_profiles(tmp_path: Path):
    config = tmp_path / "c.toml"
    config.write_text('default = "a"\n')
    with pytest.raises(ConfigError, match="no profiles"):
        load_config(config)


def test_find_rcc_dir_in_cwd(rcc_project: Path):
    assert find_rcc_dir(rcc_project) == rcc_project / ".rcc"


def test_find_rcc_dir_walks_up(rcc_project: Path):
    sub = rcc_project / "a" / "b"
    sub.mkdir(parents=True)
    assert find_rcc_dir(sub) == rcc_project / ".rcc"


def test_find_rcc_dir_returns_none_when_absent(tmp_path: Path):
    assert find_rcc_dir(tmp_path) is None


def test_resolve_profile_uses_default(rcc_project: Path):
    profile = resolve_profile(rcc_project)
    assert profile.host == "alpha.example.com"
    assert profile.remote_dir == "/srv/alpha"


def test_resolve_profile_named(tmp_path: Path):
    project = tmp_path / "p"
    project.mkdir()
    rcc = project / ".rcc"
    rcc.mkdir()
    (rcc / "config.toml").write_text(
        'default = "alpha"\n'
        '[profiles.alpha]\nhost = "a"\nremote_dir = "/a"\n'
        '[profiles.beta]\nhost = "b"\nremote_dir = "/b"\n'
    )
    profile = resolve_profile(project, profile_name="beta")
    assert profile.host == "b"


def test_resolve_profile_cli_overrides(rcc_project: Path):
    profile = resolve_profile(rcc_project, host_override="other", remote_dir_override="/x")
    assert profile.host == "other"
    assert profile.remote_dir == "/x"


def test_resolve_profile_layers_project_keys_over_user_config(
    rcc_project: Path, isolated_user_config: Path
):
    isolated_user_config.parent.mkdir(parents=True)
    isolated_user_config.write_text(
        'default = "beta"\n'
        '[profiles.beta]\nhost = "other"\n'
        '[profiles.alpha]\nhost = "shared"\nproxy_jump = "gateway"\n'
        'keep_remote = ["keep/"]\n'
        '[profiles.alpha.env]\nCOMMON = "shared"\nMODE = "default"\n'
    )
    (rcc_project / ".rcc" / "config.toml").write_text(
        'default = "alpha"\n'
        '[profiles.alpha]\nremote_dir = "/project"\nkeep_remote = []\n'
        '[profiles.alpha.env]\nMODE = "project"\n'
    )
    profile = resolve_profile(rcc_project)
    assert profile.host == "shared"
    assert profile.proxy_jump == "gateway"
    assert profile.remote_dir == "/project"
    assert profile.keep_remote == []
    assert profile.env == {"COMMON": "shared", "MODE": "project"}
    beta = resolve_profile(rcc_project, profile_name="beta", remote_dir_override="/explicit")
    assert (beta.host, beta.remote_dir) == ("other", "/explicit")


def test_resolve_profile_uses_home_config_without_xdg(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    user_config = tmp_path / ".config" / "rcc" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text('[profiles.alpha]\nhost = "a"\n')
    profile = resolve_profile(tmp_path, require_remote_dir=False)
    assert (profile.host, profile.remote_dir) == ("a", ".")
    with pytest.raises(ConfigError, match="needs a remote_dir"):
        resolve_profile(tmp_path)


def test_resolve_profile_raises_when_no_rcc_dir(tmp_path: Path):
    with pytest.raises(ConfigError, match="no .rcc/"):
        resolve_profile(tmp_path)


def test_resolve_profile_raises_on_unknown_profile(rcc_project: Path):
    with pytest.raises(ConfigError, match="gamma"):
        resolve_profile(rcc_project, profile_name="gamma")


def test_profile_exports_expected_defaults():
    profile = Profile(host="h", remote_dir="/r")
    assert profile.ssh_control_persist == "30m"
    assert profile.proxy_jump is None
    assert profile.identity_file is None
    assert profile.env == {}
    assert profile.keep_remote == []
    assert profile.tunnel is None
    assert profile.docker is None


def _rich_config(tmp_path: Path) -> Path:
    project = tmp_path / "p"
    rcc = project / ".rcc"
    rcc.mkdir(parents=True)
    (rcc / "config.toml").write_text(
        'default = "alpha"\n'
        "[profiles.alpha]\n"
        'host = "h"\n'
        'remote_dir = "/r"\n'
        'proxy_jump = "bastion"\n'
        'identity_file = "~/.ssh/id"\n'
        'keep_remote = ["logs/", "*.safetensors"]\n'
        "[profiles.alpha.env]\n"
        'TRITON_CACHE_DIR = "/scratch/cache"\n'
        "[profiles.alpha.tunnel]\n"
        "remote_port = 8080\n"
        "local_port = 18080\n"
        'remote_host = "head01"\n'
        "[profiles.alpha.docker]\n"
        'image = "pytorch/pytorch:runtime"\n'
        'workdir = "/workspace"\n'
        'gpus = "all"\n'
        'shm_size = "16g"\n'
        'network = "host"\n'
        'args = ["--ipc=host"]\n'
        "[[profiles.alpha.docker.mounts]]\n"
        'source = "{remote_dir}"\n'
        'target = "/workspace"\n'
        'mode = "rw"\n'
        "[[profiles.alpha.docker.mounts]]\n"
        'source = "/scratch/datasets"\n'
        'target = "/data"\n'
        'mode = "ro"\n'
        "[profiles.alpha.docker.env]\n"
        'HF_HOME = "/workspace/.cache/huggingface"\n'
    )
    return project


def test_load_config_parses_rich_profile(tmp_path: Path):
    project = _rich_config(tmp_path)
    cfg = load_config(project / ".rcc" / "config.toml")
    p = cfg.profiles["alpha"]
    assert p.proxy_jump == "bastion"
    assert p.identity_file == "~/.ssh/id"
    assert p.env == {"TRITON_CACHE_DIR": "/scratch/cache"}
    assert p.keep_remote == ["logs/", "*.safetensors"]
    assert isinstance(p.tunnel, TunnelSpec)
    assert p.tunnel.remote_port == 8080
    assert p.tunnel.local_port == 18080
    assert p.tunnel.remote_host == "head01"
    assert isinstance(p.docker, DockerSpec)
    assert p.docker.image == "pytorch/pytorch:runtime"
    assert p.docker.workdir == "/workspace"
    assert p.docker.gpus == "all"
    assert p.docker.shm_size == "16g"
    assert p.docker.network == "host"
    assert p.docker.args == ["--ipc=host"]
    assert p.docker.mounts[0].source == "{remote_dir}"
    assert p.docker.mounts[1].mode == "ro"
    assert p.docker.env == {"HF_HOME": "/workspace/.cache/huggingface"}


def test_load_config_tunnel_defaults_remote_host(tmp_path: Path):
    project = tmp_path / "p"
    rcc = project / ".rcc"
    rcc.mkdir(parents=True)
    (rcc / "config.toml").write_text(
        'default = "a"\n[profiles.a]\nhost="h"\nremote_dir="/r"\n'
        "[profiles.a.tunnel]\nremote_port=80\n"
    )
    p = load_config(rcc / "config.toml").profiles["a"]
    assert p.tunnel.remote_host == "localhost"
    assert p.tunnel.local_port is None


def test_load_config_rejects_non_int_tunnel_port(tmp_path: Path):
    project = tmp_path / "p"
    rcc = project / ".rcc"
    rcc.mkdir(parents=True)
    (rcc / "config.toml").write_text(
        'default = "a"\n[profiles.a]\nhost="h"\nremote_dir="/r"\n'
        '[profiles.a.tunnel]\nremote_port="x"\n'
    )
    with pytest.raises(ConfigError, match="remote_port"):
        load_config(rcc / "config.toml")


def test_load_config_rejects_bad_keep_remote(tmp_path: Path):
    project = tmp_path / "p"
    rcc = project / ".rcc"
    rcc.mkdir(parents=True)
    (rcc / "config.toml").write_text(
        'default = "a"\n[profiles.a]\nhost="h"\nremote_dir="/r"\nkeep_remote=42\n'
    )
    with pytest.raises(ConfigError, match="keep_remote"):
        load_config(rcc / "config.toml")


def test_profile_to_dict_is_json_serializable(tmp_path: Path):
    import json

    project = _rich_config(tmp_path)
    p = resolve_profile(project)
    d = profile_to_dict(p)
    # round-trips through json without error and keeps nested structure
    round_tripped = json.loads(json.dumps(d))
    assert round_tripped["env"] == {"TRITON_CACHE_DIR": "/scratch/cache"}
    assert round_tripped["tunnel"]["remote_port"] == 8080
    assert round_tripped["docker"]["mounts"][0]["target"] == "/workspace"


def test_get_field_top_level_and_dotted(tmp_path: Path):
    project = _rich_config(tmp_path)
    p = resolve_profile(project)
    assert get_field(p, "host") == "h"
    assert get_field(p, "remote_dir") == "/r"
    assert get_field(p, "proxy_jump") == "bastion"
    assert get_field(p, "env.TRITON_CACHE_DIR") == "/scratch/cache"
    assert get_field(p, "tunnel.remote_port") == "8080"
    assert get_field(p, "tunnel.remote_host") == "head01"
    assert get_field(p, "docker.image") == "pytorch/pytorch:runtime"
    assert get_field(p, "docker.workdir") == "/workspace"
    assert get_field(p, "docker.env.HF_HOME") == "/workspace/.cache/huggingface"


def test_get_field_missing_key_errors(tmp_path: Path):
    project = _rich_config(tmp_path)
    p = resolve_profile(project)
    with pytest.raises(ConfigError, match="env has no key"):
        get_field(p, "env.NOPE")
    with pytest.raises(ConfigError, match="unknown profile field"):
        get_field(p, "nope")
    with pytest.raises(ConfigError, match="structured"):
        get_field(p, "docker.mounts")


def test_load_config_parses_docker_container_profile(tmp_path: Path):
    project = tmp_path / "p"
    rcc = project / ".rcc"
    rcc.mkdir(parents=True)
    (rcc / "config.toml").write_text(
        'default = "a"\n[profiles.a]\nhost="h"\nremote_dir="/r"\n'
        "[profiles.a.docker]\n"
        'container = "devbox"\n'
        'workdir = "/workspace"\n'
    )
    p = load_config(rcc / "config.toml").profiles["a"]
    assert p.docker is not None
    assert p.docker.container == "devbox"
    assert p.docker.image is None


def test_load_config_rejects_docker_image_and_container(tmp_path: Path):
    project = tmp_path / "p"
    rcc = project / ".rcc"
    rcc.mkdir(parents=True)
    (rcc / "config.toml").write_text(
        'default = "a"\n[profiles.a]\nhost="h"\nremote_dir="/r"\n'
        "[profiles.a.docker]\n"
        'image = "img"\n'
        'container = "devbox"\n'
    )
    with pytest.raises(ConfigError, match="mutually exclusive"):
        load_config(rcc / "config.toml")


def test_load_config_rejects_bad_docker_mount_mode(tmp_path: Path):
    project = tmp_path / "p"
    rcc = project / ".rcc"
    rcc.mkdir(parents=True)
    (rcc / "config.toml").write_text(
        'default = "a"\n[profiles.a]\nhost="h"\nremote_dir="/r"\n'
        "[profiles.a.docker]\n"
        'image = "img"\n'
        "[[profiles.a.docker.mounts]]\n"
        'source = "/x"\n'
        'target = "/y"\n'
        'mode = "bad"\n'
    )
    with pytest.raises(ConfigError, match="ro"):
        load_config(rcc / "config.toml")
