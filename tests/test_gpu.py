from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from rcc import gpu
from rcc.cli import app
from rcc.config import Profile

runner = CliRunner()


@pytest.fixture
def mock_nvidia(tmp_path: Path, monkeypatch) -> Path:
    smi = tmp_path / "nvidia-smi"
    smi.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "path = pathlib.Path(os.environ['MOCK_GPU_STATE'])\n"
        "if sys.argv[1].startswith('--query-gpu='):\n"
        "    print('0, GPU-a, 4\\n1, GPU-b, 16')\n"
        "else:\n"
        "    value = path.read_text() if path.exists() else ''\n"
        "    print(value, end='')\n"
        "    with path.with_suffix('.queries').open('a') as queries:\n"
        "        print('busy' if value else 'idle', file=queries)\n"
    )
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    state = tmp_path / "gpu-state"
    monkeypatch.setenv("MOCK_GPU_STATE", str(state))
    return state


def _gpu_run(tmp_path, command, *, indices, checks=0):
    return gpu.run_argv(
        Profile(host="host", remote_dir=str(tmp_path)),
        command,
        script=None,
        work_dir=str(tmp_path),
        docker=None,
        tty=False,
        session="sweep",
        indices=indices,
        lock=str(tmp_path / "gpu{N}.lock"),
        idle_checks=checks,
        idle_interval=0.02,
        idle_memory_mib=20,
    )


def _inventory(tmp_path):
    def local_capture(_profile, argv, *, cwd):
        return subprocess.run(argv, capture_output=True, text=True, timeout=5)

    with patch("rcc.gpu.run_remote_capture", side_effect=local_capture):
        return gpu.list_gpus(Profile("host", str(tmp_path)), str(tmp_path / "gpu{N}.lock"))


def _await_file(path):
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def test_gpu_claim_blocks_overlap_and_is_visible(tmp_path, mock_nvidia):
    ready, done = (tmp_path / name for name in ("ready", "done"))
    holder = _gpu_run(
        tmp_path,
        [
            "bash",
            "-c",
            'touch "$1"; while [[ ! -e "$2" ]]; do sleep .02; done',
            "bash",
            str(ready),
            str(done),
        ],
        indices=[1],
    )
    owner = subprocess.Popen(holder)
    try:
        _await_file(ready)
        rows = _inventory(tmp_path)
        assert rows[0]["locked"] is False
        assert rows[1]["locked"] is True
        claim = rows[1]["claim"]
        assert claim["session"] == "sweep" and claim["owner"] and claim["started_at"]
        competing = subprocess.run(
            _gpu_run(tmp_path, ["false"], indices=[1, 0]),
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert competing.returncode != 0
        assert "already locked" in competing.stderr
        assert _inventory(tmp_path)[0]["locked"] is False
    finally:
        done.touch()
        assert owner.wait(timeout=5) == 0
    assert _inventory(tmp_path)[1]["claim"] is None


def test_gpu_waits_for_apps_before_launching(tmp_path, mock_nvidia):
    mock_nvidia.write_text("GPU-a, 123\n")
    command = _gpu_run(
        tmp_path, ["bash", "-c", 'echo "$CUDA_VISIBLE_DEVICES"'], indices=[0], checks=2
    )
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as child:
        queries = mock_nvidia.with_suffix(".queries")
        _await_file(queries)
        assert child.poll() is None
        mock_nvidia.write_text("")
        out, _ = child.communicate(timeout=5)
    assert child.returncode == 0
    assert out.strip() == "GPU-a"
    assert queries.read_text().splitlines()[-2:] == ["idle", "idle"]


def test_cli_managed_docker_detach_uses_gpu_create(rcc_project, monkeypatch):
    monkeypatch.chdir(rcc_project)
    with patch("rcc.commands.run.bg.start", return_value="sweep-1") as start:
        result = runner.invoke(
            app,
            [
                "run",
                "--detach",
                "--name",
                "sweep.1",
                "--docker-image",
                "image:cuda",
                "--gpus",
                "1,0",
                "--wait-idle-gpus",
                "--idle-checks",
                "2",
                "--idle-interval",
                "0.5",
                "--",
                "echo",
                "a b",
            ],
        )
    assert result.exit_code == 0, result.output
    kwargs = json.loads(start.call_args.args[2][-1])
    assert kwargs["indices"] == [1, 0]
    assert kwargs["session"] == "sweep-1"
    assert kwargs["idle_checks"] == 2
    assert kwargs["idle_interval"] == 0.5
    command = kwargs["command"]
    assert command[:2] == ["docker", "create"]
    assert command[command.index("--gpus") + 1] == '"device=1,0"'
    assert command[-3:] == ["image:cuda", "echo", "a b"]


def test_cli_rejects_denied_gpu_indices(rcc_project, monkeypatch):
    monkeypatch.chdir(rcc_project)
    path = rcc_project / ".rcc" / "config.toml"
    path.write_text(path.read_text() + "deny_gpus = [2]\n")
    result = runner.invoke(app, ["run", "--detach", "--gpus", "0,2", "--", "true"])
    assert result.exit_code != 0
    assert "profile denies GPU indices: 2" in result.output
