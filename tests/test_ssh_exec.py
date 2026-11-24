from __future__ import annotations

import socket
import subprocess
from unittest.mock import patch

import pytest

from rcc.config import DockerSpec, Profile
from rcc.errors import MissingDependencyError, RemoteError
from rcc.slurm import ensure_slurm
from rcc.ssh import (
    build_rsync_e_string,
    build_ssh_local_forward_args,
    ensure_remote_dir,
    mux_check,
    mux_exit,
    run_remote,
    run_remote_capture,
)


def make_profile(**kw) -> Profile:
    base = {"host": "myhost", "remote_dir": "/srv/app"}
    base.update(kw)
    return Profile(**base)


@pytest.fixture(autouse=True)
def _force_primary(monkeypatch):
    monkeypatch.setattr("rcc.ssh.shutil.which", lambda name: f"/usr/bin/{name}")


@pytest.fixture
def fake_run():
    with patch("rcc.ssh.subprocess.run") as mocked:
        yield mocked


def test_run_remote_invokes_ssh(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    code = run_remote(make_profile(), ["ls"], tty=False)
    assert code == 0
    argv = fake_run.call_args.args[0]
    assert argv[0] == "ssh"
    assert argv[-2] == "myhost"
    assert argv[-1].startswith("bash -lc ")
    assert "ls" in argv[-1]


def test_reuse_user_controlmaster(fake_run, tmp_path):
    control = tmp_path / "my socket"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(control))

        def ssh(argv, **kwargs):
            if "-G" in argv:
                return subprocess.CompletedProcess(argv, 0, f"controlpath {control}\n")
            return subprocess.CompletedProcess(argv, 0)

        fake_run.side_effect = ssh
        profile = make_profile(proxy_jump="gateway")
        assert run_remote(profile, ["ls"]) == 0
        probe, check, command = [call.args[0] for call in fake_run.call_args_list]
        assert "-G" in probe
        assert check[-3:] == ["-O", "check", profile.host]
        assert command[-2] == profile.host
        for argv in (probe, check, command):
            assert "ProxyJump=gateway" in argv
            assert "ControlMaster=no" in argv
            assert not any(arg.startswith("ControlPath=") for arg in argv)
        assert "ControlMaster=no" in build_rsync_e_string(profile)
        assert "ControlMaster=no" in build_ssh_local_forward_args(
            profile, local_port=9000, remote_port=8080
        )
        assert mux_check(profile, include_user=True)
        mux_exit(profile)
        assert "ControlPath=~/.ssh/controlmasters/%C" in fake_run.call_args.args[0]


def test_stale_user_socket_falls_back_to_rcc(fake_run, tmp_path):
    control = tmp_path / "stale"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(control))
    fake_run.side_effect = [
        subprocess.CompletedProcess([], 0, f"controlpath {control}\n"),
        subprocess.CompletedProcess([], 255),
        subprocess.CompletedProcess([], 0),
    ]
    assert run_remote(make_profile(), ["ls"]) == 0
    assert "ControlPath=~/.ssh/controlmasters/%C" in fake_run.call_args.args[0]


@pytest.mark.parametrize(
    ("exit_code", "diagnostic", "message"),
    [
        (255, "myhost: Permission denied (keyboard-interactive).", "SSH authentication"),
        (255, "ssh: connect to host myhost port 22: Connection refused", "SSH authentication"),
        (1, "myhost: Permission denied (publickey).", "SSH authentication"),
        (1, "error: remote directory does not exist: /srv/app", "could not check Slurm"),
    ],
)
def test_slurm_probe_preserves_connection_or_remote_error(fake_run, exit_code, diagnostic, message):
    fake_run.return_value = subprocess.CompletedProcess([], exit_code, "", diagnostic)
    with pytest.raises(RemoteError, match=message) as err:
        ensure_slurm(make_profile())
    assert err.value.exit_code == exit_code
    assert diagnostic in str(err.value)


def test_run_remote_quotes_remote_script_as_single_ssh_command(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    run_remote(make_profile(remote_dir="/srv/my app"), ["printf", "semi;colon"], tty=False)
    argv = fake_run.call_args.args[0]
    assert argv[-2] == "myhost"
    assert argv[-1].startswith("bash -lc ")
    assert "semi;colon" in argv[-1]


def test_run_remote_shell_mode_passes_script_unquoted(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    run_remote(make_profile(), [], script="squeue -u $USER | head", tty=False)
    joined = fake_run.call_args.args[0][-1]
    # $USER and the pipeline survive because the snippet is not per-token quoted
    assert "$USER" in joined
    assert " | " in joined
    assert "'squeue -u $USER | head'" not in joined


def test_run_remote_propagates_nonzero_exit(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=42)
    assert run_remote(make_profile(), ["false"]) == 42


def test_run_remote_tty_flag(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    run_remote(make_profile(), ["ls"], tty=True)
    assert "-t" in fake_run.call_args.args[0]


def test_ensure_remote_dir_mkdirs(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    ensure_remote_dir(make_profile())
    assert "mkdir -p -- /srv/app" in fake_run.call_args.args[0][-1]


def test_ensure_remote_path_mkdirs_arbitrary(fake_run):
    from rcc.ssh import ensure_remote_path

    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    ensure_remote_path(make_profile(), "/srv/app/jobs/sweep")
    assert "mkdir -p -- /srv/app/jobs/sweep" in fake_run.call_args.args[0][-1]


def test_ensure_remote_path_raises_on_failure(fake_run):
    from rcc.ssh import ensure_remote_path

    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
    with pytest.raises(RemoteError):
        ensure_remote_path(make_profile(), "/srv/app/sub")


def test_ensure_remote_dir_raises_on_failure(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
    with pytest.raises(RemoteError):
        ensure_remote_dir(make_profile())


def test_run_remote_passes_env_and_cwd(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    run_remote(make_profile(), ["env"], env={"X": "1"}, cwd="/srv/app/sub")
    joined = fake_run.call_args.args[0][-1]
    assert "export X=1;" in joined
    assert "cd -- /srv/app/sub" in joined


def test_run_remote_threads_docker_target(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    run_remote(
        make_profile(),
        ["python", "train.py"],
        env={"REMOTE": "1"},
        docker=DockerSpec(image="img", env={"E": "2"}),
    )
    joined = fake_run.call_args.args[0][-1]
    assert "export REMOTE=1;" in joined
    assert "docker run --rm" in joined
    assert "--env E=2" in joined
    assert "img python train.py" in joined


def test_run_remote_capture_threads_docker_target(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ok", stderr=""
    )
    result = run_remote_capture(make_profile(), ["python", "-V"], docker=DockerSpec(image="img"))
    assert result.stdout == "ok"
    joined = fake_run.call_args.args[0][-1]
    assert "docker run --rm" in joined
    assert "img python -V" in joined


def test_mux_check_returns_true_when_open(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    assert mux_check(make_profile()) is True


def test_mux_check_returns_false_when_closed(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
    assert mux_check(make_profile()) is False


def test_mux_exit_issues_exit_command(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    mux_exit(make_profile())
    argv = fake_run.call_args.args[0]
    assert "-O" in argv and "exit" in argv


def test_run_remote_capture_returns_completed_process(fake_run):
    fake_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ok", stderr=""
    )
    result = run_remote_capture(make_profile(), ["whoami"])
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0
    assert result.stdout == "ok"
    argv = fake_run.call_args.args[0]
    assert argv[0] == "ssh"
    assert "whoami" in argv[-1]
    assert "capture_output" in fake_run.call_args.kwargs


def test_run_remote_capture_uses_paramiko_when_ssh_missing(monkeypatch):
    monkeypatch.setattr("rcc.ssh.shutil.which", lambda name: None)
    cp = subprocess.CompletedProcess(args=[], returncode=0, stdout="x", stderr="")
    monkeypatch.setattr("rcc.ssh._paramiko_run_remote_capture", lambda *a, **k: cp)
    assert run_remote_capture(make_profile(), ["ls"]).stdout == "x"


def test_run_remote_uses_paramiko_when_ssh_missing(monkeypatch):
    monkeypatch.setattr("rcc.ssh.shutil.which", lambda name: None)
    called = {}

    def fake_fallback(profile, argv, *, script, tty, env=None, cwd=None, docker=None):
        called["profile"] = profile
        called["argv"] = argv
        called["script"] = script
        called["tty"] = tty
        called["docker"] = docker
        return 0

    monkeypatch.setattr("rcc.ssh._paramiko_run_remote", fake_fallback)
    assert run_remote(make_profile(), ["ls"], tty=True) == 0
    assert called["argv"] == ["ls"]
    assert called["script"] is None
    assert called["tty"] is True
    assert called["docker"] is None


def test_mux_check_returns_false_when_ssh_missing(monkeypatch):
    monkeypatch.setattr("rcc.ssh.shutil.which", lambda name: None)
    assert mux_check(make_profile()) is False


def test_mux_exit_raises_when_ssh_missing(monkeypatch):
    monkeypatch.setattr("rcc.ssh.shutil.which", lambda name: None)
    with pytest.raises(MissingDependencyError):
        mux_exit(make_profile())
