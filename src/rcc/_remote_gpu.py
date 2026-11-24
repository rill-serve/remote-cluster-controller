"""Standalone GPU supervisor sent to the remote host as Python source."""

import contextlib
import csv
import datetime
import fcntl
import glob
import json
import os
import pwd
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time


def _query(option):
    result = subprocess.run(
        ["nvidia-smi", option, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = (line for line in result.stdout.splitlines() if line.strip())
    return ([field.strip() for field in row] for row in csv.reader(lines))


def _snapshot():
    by_uuid = {}
    for index, uuid, memory in _query("--query-gpu=index,uuid,memory.used"):
        by_uuid[uuid] = {
            "index": int(index),
            "uuid": uuid,
            "memory_used_mib": int(memory) if memory.isdecimal() else None,
            "compute_pids": [],
        }
    for uuid, pid in _query("--query-compute-apps=gpu_uuid,pid"):
        if uuid not in by_uuid:
            raise ValueError("nvidia-smi returned a compute process on an unknown GPU")
        by_uuid[uuid]["compute_pids"].append(int(pid))
    return {row["index"]: row for row in by_uuid.values()}


def _selected_snapshot(indices):
    snapshot = _snapshot()
    missing = sorted(set(indices) - snapshot.keys())
    if missing:
        raise ValueError(f"unknown GPU indices: {', '.join(map(str, missing))}")
    return [snapshot[index] for index in indices]


def _is_idle(rows, threshold):
    if any(row["memory_used_mib"] is None for row in rows):
        raise ValueError("nvidia-smi did not report memory use for a selected GPU")
    return all(not row["compute_pids"] and row["memory_used_mib"] <= threshold for row in rows)


def _path(template, index):
    path = template.replace("{N}", str(index))
    return os.path.join(os.path.realpath(os.path.dirname(path)), os.path.basename(path))


def _open_lock(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"GPU lock must be a regular file: {path}")
    return fd


def _open_or_create_lock(path):
    try:
        return _open_lock(path)
    except FileNotFoundError:
        # O_EXCL never follows a symlink or attempts to open another user's file for writing.
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            pass
        else:
            try:
                os.fchmod(fd, 0o644)
            finally:
                os.close(fd)
        return _open_lock(path)


def _try_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


@contextlib.contextmanager
def _claim(paths):
    with contextlib.ExitStack() as stack:
        fds = []
        for path in paths:
            fd = _open_or_create_lock(path)
            fds.append(fd)
            stack.callback(os.close, fd)
            if not _try_lock(fd):
                stack.close()
                yield None
                return
        yield fds


@contextlib.contextmanager
def _ownership(paths, options):
    data = {
        "session": options["session"],
        "image": options["image"],
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "pid": os.getpid(),
    }
    with contextlib.ExitStack() as stack:
        files = []
        for path in paths:
            fd, metadata = tempfile.mkstemp(
                dir=os.path.dirname(path), prefix=os.path.basename(path) + ".owner."
            )
            stack.callback(os.close, fd)
            stack.callback(os.unlink, metadata)
            fcntl.flock(fd, fcntl.LOCK_EX)
            files.append(fd)

        def publish(**extra):
            data.update(extra)
            line = (json.dumps(data) + "\n").encode("utf-8")
            for fd in files:
                os.write(fd, line)
                os.fchmod(fd, 0o644)

        publish()
        yield publish


def _lock_info(path):
    try:
        fd = _open_lock(path)
    except FileNotFoundError:
        return False, None
    try:
        if _try_lock(fd):
            return False, None
    finally:
        os.close(fd)

    for metadata in glob.glob(glob.escape(path) + ".owner.*"):
        try:
            fd = _open_lock(metadata)
            with os.fdopen(fd, encoding="utf-8") as owner:
                if _try_lock(owner.fileno()):
                    continue  # A stale record survived an unclean exit.
                lines = owner.readlines()
                if lines:
                    record = json.loads(lines[-1])
                    uid = os.fstat(owner.fileno()).st_uid
                    try:
                        record["owner"] = pwd.getpwuid(uid).pw_name
                    except KeyError:
                        record["owner"] = str(uid)
                    return True, record
        except (OSError, ValueError, TypeError):
            continue  # The holder may have exited or started writing after the lock probe.
    return True, None


class _Interrupt:
    signum = None

    def __call__(self, number, _frame):
        self.signum = self.signum or number

    def sleep(self, seconds):
        deadline = time.monotonic() + seconds
        while self.signum is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.25))


def _wait_process(process, interrupt, on_interrupt, *, retry_interrupt=False):
    next_signal = 0
    while True:
        if (
            interrupt.signum is not None
            and time.monotonic() >= next_signal
            and process.poll() is None
        ):
            on_interrupt()
            next_signal = time.monotonic() + (1 if retry_interrupt else float("inf"))
        try:
            return process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            continue


def _docker_cleanup(container, interrupt):
    # An attached Docker CLI can exit while its container is still running.
    # Keep the GPU claim until the daemon confirms the container stopped.
    warned = False
    while True:
        state = subprocess.run(
            [
                "docker",
                "container",
                "ls",
                "--all",
                "--filter",
                f"id={container}",
                "--format",
                "{{.State}}",
            ],
            text=True,
            capture_output=True,
            start_new_session=True,
        )
        if state.returncode == 0:
            if state.stdout.strip() in {"", "created", "exited", "dead"}:
                return
            if interrupt.signum is not None:
                _stop_container(container)
        elif not warned:
            with contextlib.suppress(OSError):
                print(
                    "Cannot verify Docker stopped. Holding GPU locks while retrying.",
                    file=sys.stderr,
                )
            warned = True
        time.sleep(1)


def _stop_container(container):
    subprocess.run(
        ["docker", "stop", "--time", "5", container],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _run_docker(options, interrupt, publish):
    # Shield create from a terminal hangup so we can always collect the ID of a created container.
    created = subprocess.run(
        options["command"], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    if created.returncode:
        return created.returncode
    container = created.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", container):
        raise ValueError("docker create did not return a container ID")
    if interrupt.signum is not None:
        subprocess.run(["docker", "rm", container], start_new_session=True)
        return 128 + interrupt.signum

    try:
        publish(container=container)
        attach = "-ai" if options["tty"] else "-a"
        with subprocess.Popen(
            ["docker", "start", attach, container], start_new_session=True
        ) as process:
            return _wait_process(
                process, interrupt, lambda: _stop_container(container), retry_interrupt=True
            )
    finally:
        _docker_cleanup(container, interrupt)


def _run_host(options, selected, fds, interrupt):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(row["uuid"] for row in selected))
    with subprocess.Popen(options["command"], env=env, pass_fds=tuple(fds)) as process:
        return _wait_process(process, interrupt, lambda: process.send_signal(interrupt.signum))


def _run(options):
    interrupt = _Interrupt()
    for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupt)
    indices = options["indices"]
    paths = [_path(options["lock"], index) for index in sorted(indices)]
    checks = options["idle_checks"]
    consecutive = 0
    while interrupt.signum is None:
        with _claim(paths) as fds:
            if fds is None:
                if not checks:
                    raise ValueError("one or more selected GPUs are already locked")
                consecutive = 0
            else:
                selected = _selected_snapshot(indices)
                if not checks or _is_idle(selected, options["idle_memory_mib"]):
                    consecutive += 1
                    if consecutive >= checks and interrupt.signum is None:
                        with _ownership(paths, options) as publish:
                            if options["image"]:
                                return _run_docker(options, interrupt, publish)
                            return _run_host(options, selected, fds, interrupt)
                else:
                    consecutive = 0
        interrupt.sleep(options["idle_interval"])
    return 128 + interrupt.signum


def _list(options):
    rows = []
    for index, row in sorted(_snapshot().items()):
        locked, claim = _lock_info(_path(options["lock"], index))
        rows.append(dict(row, locked=locked, claim=claim, denied=index in options["deny_gpus"]))
    print(json.dumps(rows))
    return 0


def main():
    mode, raw = sys.argv[1:]
    try:
        options = json.loads(raw)
        code = _run(options) if mode == "run" else _list(options)
        return 128 - code if code < 0 else code
    except subprocess.CalledProcessError as exc:
        print(f"GPU check failed: {exc.stderr.strip() or exc}", file=sys.stderr)
        return exc.returncode
    except (OSError, ValueError) as exc:
        print(f"GPU coordination failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
