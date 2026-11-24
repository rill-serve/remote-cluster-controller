# rcc

Remote cluster controller. Generalizes the classic `ssh host "cd dir && cmd"` +
`rsync project/ host:dir/` workflow into a small CLI with per-project config.

## Install

```bash
uv tool install remote-cluster-controller
# or
pipx install remote-cluster-controller
```

Both install the `rcc` command.

## Quickstart

```bash
cd my-project
rcc init
# edit .rcc/config.toml to set host and remote_dir
rcc push --dry-run
rcc push
rcc run -- nvidia-smi
rcc run -t -- htop
rcc run -s 'squeue -u $USER | head'   # shell snippet: pipelines, $vars survive
rcc run --env EPOCHS=10 -- python train.py
rcc run --docker -- python train.py   # run through the profile's remote Docker target
rcc shell
rcc pull jobs/results/                 # fetch just that subtree (bypasses rccignore)
rcc status
rcc close
```

## Configuration

`.rcc/config.toml` (per project, gitignored):

```toml
default = "myhost"

[profiles.myhost]
host = "myhost"
remote_dir = "/abs/path/on/remote"
# Bastion hop / explicit key (encapsulate instead of an SSH-config dance):
# proxy_jump = "bastion.example.com"
# identity_file = "~/.ssh/id_ed25519"

# Remote-owned paths to protect from --delete/--mirror (rccignore-with-teeth):
# keep_remote = ["logs/", "cache/", "*.safetensors", "last_service.env"]

# Optional coordination settings for a shared GPU host.
# gpu_lock = "/tmp/rcc-gpu{N}.lock"
# deny_gpus = [2]

# Per-profile env defaults, honored by `rcc run` and `rcc job submit`:
[profiles.myhost.env]
TRITON_CACHE_DIR = "/scratch/cache"

# Default port-forward for `rcc tunnel`:
# [profiles.myhost.tunnel]
# remote_port = 8080
# local_port = 18080        # optional; defaults to remote_port
# remote_host = "localhost" # optional; set to a compute node to reach a service there

# Optional Docker target for `rcc run --docker`.
# Docker runs on the SSH remote host, so images and mount sources are resolved
# there, not on your laptop.
[profiles.myhost.docker]
image = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
workdir = "/workspace"
gpus = "all"
shm_size = "16g"
network = "host"

[[profiles.myhost.docker.mounts]]
source = "{remote_dir}"
target = "/workspace"
mode = "rw"

[[profiles.myhost.docker.mounts]]
source = "/scratch/datasets"
target = "/data"
mode = "ro"

[profiles.myhost.docker.env]
HF_HOME = "/workspace/.cache/huggingface"
```

Per-command overrides: `--profile`, `--host`, `--remote-dir`.
Excludes: edit `.rcc/rccignore` (gitignore syntax).

### SSH connections

RCC reuses an active OpenSSH ControlMaster from your SSH configuration when the
profile's host or alias resolves to its socket. On a host that requires 2FA,
open the configured master with `ssh <alias>` and use that alias as the
profile's `host`. RCC uses its own control socket when the configured master
is unavailable. `rcc status` checks both sockets, and `rcc close` closes the
RCC-managed master.

### Machine-readable config

`rcc config` prints the resolved profile. For automation:

```bash
rcc config --json                    # stable JSON object of the resolved profile
rcc config --get host                # single value, unquoted
rcc config --get remote_dir
rcc config --get env.TRITON_CACHE_DIR
rcc config --get tunnel.remote_port
rcc config --get docker.image
rcc config --get docker.workdir
rcc config --get docker.env.HF_HOME
```

This lets wrappers drop fragile `sed`/regex parsing of free text.

## Sync: push / pull

```bash
rcc push                             # whole project → remote
rcc push jobs/sweep                  # just a subpath (bypasses rccignore)
rcc pull                             # whole remote → project
rcc pull jobs/sweep/                 # fetch a subtree even if it's in rccignore
rcc pull jobs/sweep/ ./out/          # ...reconstructed under ./out/
rcc push --no-ignore                 # bypass rccignore for a whole-project transfer
rcc push --include '*.bin'           # extra include globs
```

### Output: quiet by default (issue #6)

`push`/`pull` no longer dump rsync's progress bar and stats block at you. A
real transfer prints a single summary line:

```
$ rcc push
Pushed 42 files (12.34MB sent) → alpha:/srv/app/
$ rcc pull
Pulled 1 file (35B received) → alpha:/srv/app/
```

Pass `-v`/`--verbose` (a root flag, before the subcommand) to see everything
rsync prints — the live progress bar, per-file list, and full stats:

```
$ rcc -v push
sending incremental file list
...
```

`--dry-run` output is unaffected (it was already concise). On failure the rsync
stderr tail is surfaced so you don't need `--verbose` just to see *why*.

### Deletion safety

The default is **non-destructive**. There are two distinct deletion modes:

- `--delete` — bounded sync: deletes remote/local files *inside the non-ignored
  transfer scope* that have vanished from the source. Excluded files are spared.
- `--mirror` — **dangerous** full mirror: also removes rccignore-excluded files
  (`--delete-excluded`). Use this only when you truly want local→remote mirroring.

`--keep-remote GLOB` (repeatable) and the profile-level `keep_remote = [...]`
list protect job-owned paths — `logs/`, `cache/`, `*.safetensors`,
`last_service.env` — from **both** modes. The guard rail lives in the tool, not
your memory.

```bash
rcc push --delete                     # safe bounded sync
rcc push --mirror --keep-remote 'logs/'   # full mirror, but never touch logs/
```

### Clear dry-runs

`--dry-run` no longer buries deletions in a wall of `f.f.....` lines. Deletions
get their own section so a destructive transfer can't hide:

```
$ rcc pull --delete --dry-run
Dry run for /srv/app/
Would DELETE:
  - jobs/old.txt
  - cache/v3/
Would RECEIVE (download):
  + jobs/new.txt
```

## Running commands: run

```bash
rcc run -- nvidia-smi                 # exec-style: tokens passed verbatim
rcc run -s 'a && b | c'              # shell snippet: pipelines, $vars, quotes survive
rcc run --env EPOCHS=10 --env GPU=0 -- python train.py   # remote env (repeatable)
rcc run --env-file .env -- python serve.py               # load KEY=VAL lines
rcc run --cwd /scratch/run -- bash job.sh                # override working dir
rcc run -t -- htop                    # allocate a PTY
```

`--env` layers on top of the profile's `[env]` defaults, removing the
injection-fragile `KEY=VAL cmd` hand-quoting wrappers used to need.

### Docker execution on the remote host

`rcc run --docker` wraps the same remote command in `docker run` or
`docker exec` on the SSH remote host. This keeps the daily command terse while
the stable image, mounts, container workdir, GPU flags, and container env live in
`.rcc/config.toml`.

```bash
rcc run --docker -- python train.py --epochs 10
rcc run --docker -s 'python train.py 2>&1 | tee train.log'
rcc run --docker-image my-debug-image -- bash
rcc run --docker-container devbox --docker-workdir /workspace -- python -V
```

With the profile above, `rcc run --docker -- python train.py` executes roughly:

```bash
ssh myhost 'cd /abs/path/on/remote && docker run --rm \
  --gpus all \
  --shm-size 16g \
  --network host \
  --mount type=bind,src=/abs/path/on/remote,dst=/workspace \
  --mount type=bind,src=/scratch/datasets,dst=/data,readonly \
  --workdir /workspace \
  --env TRITON_CACHE_DIR=/scratch/cache \
  --env HF_HOME=/workspace/.cache/huggingface \
  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  python train.py'
```

Important semantics:

- The Docker image is resolved by Docker on the remote machine. If the image
  only exists locally, push it to a registry or load it on the remote first.
- Docker mount sources are remote-host paths. `{remote_dir}` expands to the
  resolved profile `remote_dir`; `{cwd}` expands to the effective remote working
  directory.
- Profile `[env]` values are still exported in the remote shell and are also
  passed into the container. `[docker.env]` is container-only and wins over
  profile env. CLI `--env` wins over profile env, and `--docker-env` wins last
  inside the container.
- `--docker-image` and `--docker-container` are mutually exclusive. Image mode
  uses `docker run --rm`; container mode uses `docker exec`.
- `--detach --docker` accepts image targets with explicit `--gpus` indices.

For one-off overrides:

```bash
rcc run --docker-image img \
  --docker-workdir /app \
  --docker-mount '{remote_dir}:/app:ro' \
  --docker-gpus all \
  -- pytest
```

### Streaming + captured output together (tee)

A long-standing snag for wrappers: plain streaming shows output live but
discards the text; plain capture buffers everything so the call looks hung.
`--result-json PATH` tees — output streams to your terminal *and* rcc writes a
structured result to `PATH` on exit:

```bash
rcc run --result-json /tmp/r.json -- srun python train.py   # live output now
# then: {"returncode": 0, "stdout": "...", "stderr": "...", "command": [...]}
```

A wrapper using `subprocess.run(argv)` (streaming) gets the live output, then
reads `/tmp/r.json` for the authoritative exit code + full captured text.

### Detached runs for non-SLURM hosts (`--detach` + `rcc bg`)

`rcc job` is Slurm-only; on a plain SSH host there used to be no built-in way
to launch a long command detached and reattach. Now there is — backed by tmux,
so it survives disconnects:

```bash
rcc run --detach --name sweep -- python train.py   # launch in a tmux session
rcc bg ps                                          # list rcc-launched sessions
rcc bg logs sweep -f                               # tail the captured log
rcc bg attach sweep                                # attach to the live session
rcc bg wait sweep                                  # block until it exits (exit code)
rcc bg stop sweep                                  # kill it
```

State lives under `remote_dir/.rcc-runs/<name>.{log,status}`, so it survives
disconnects and can be pulled back. Requires `tmux` on the remote (auto-detected;
exit `127` with a hint if missing). The `--name` is optional (auto-generated);
`logs`/`attach`/`wait`/`stop` default to the sole running session if you omit it.
This removes the hand-rolled tmux orchestration every non-SLURM consumer needed.

### Shared GPU hosts

```bash
rcc run --docker --gpus 0,1 --wait-idle-gpus -- python train.py
rcc run --detach --name sweep --gpus 0,1 --wait-idle-gpus \
  --idle-checks 3 --idle-interval 120 --idle-memory-mib 256 -- python train.py
rcc gpus
rcc gpus --json
```

`--gpus` selects host GPU indices from `nvidia-smi` and locks each selected
GPU for the run. `--wait-idle-gpus` polls until all selected GPUs are unlocked,
have no compute processes, and are at or below the memory threshold for the
configured number of consecutive checks. The defaults are three checks,
120 seconds between checks, and 256 MiB per GPU. A run using `--gpus` alone
fails if a selected GPU is already locked.

Users coordinating on the same host need a common lock path. The default is
`/tmp/rcc-gpu{N}.lock`, where `{N}` is the GPU index. Set `gpu_lock` in the
profile or pass `--gpu-lock` to both `run` and `gpus` for a custom path.
`rcc gpus` reports device usage and active claims, including owner, session,
image, and start time when available. Locks coordinate launches that use the
same lock path. The idle checks also detect compute processes from other
launchers.

GPU coordination requires `python3` and `nvidia-smi` on the remote host.
Managed Docker runs create an image-based container for the selected GPUs.
The profile's `deny_gpus` list reserves indices from GPU-managed runs.

## Port-forwarding: tunnel

```bash
rcc tunnel                            # uses the profile [tunnel] defaults
rcc tunnel --remote-port 8080         # ...or specify explicitly
rcc tunnel --remote-port 8080 --remote-host head01 --local-port 18080
```

`rcc tunnel` opens a local port-forward using the SSH ControlMaster for the
profile. Ctrl-C closes the tunnel.

## Slurm jobs (`rcc job`)

For HPC login nodes running Slurm, `rcc job` wraps the common verbs so you
never have to shell-quote a `--format=` value, a `$USER`, or a pipeline (the
friction that motivated this command — see issue #1).

```bash
rcc job submit train.sh --extra-env EPOCHS=10   # sbatch, prints the JOBID
rcc job submit train.sh -W --dependency afterok:524614   # block until done + chain after a prior job
rcc job list                                     # squeue for your user
rcc job list --json                              # ...as structured records for wrappers
rcc job list --start --json                      # include pending-job start estimates
rcc job update 524615 --dependency afterok:524614  # update an existing job's dependency
rcc job status 524614                            # sacct -j <id> (fixed format)
rcc job status 524614 --json                     # ...as structured records (main + steps)
rcc job tail 524614 -f                           # tail -f slurm-524614.out
rcc job wait 524614                              # poll until done, exit with job's code
rcc job wait 524614 --on RUNNING                 # ...or until it reaches a state
rcc job cancel 524614                            # scancel
```

Notes:

- Slurm commands check for `sbatch` and exit `127` when it is missing. SSH
  connection failures show the SSH diagnostic. `job tail` reads the log
  directly.
- The `job list` table and JSON output contain the same jobs. JSON includes
  `Reason`, `TimeLimit`, `StartTime`, `Dependency`, `Account`, `SubmitTime`,
  and `NumNodes`. `--start` adds start estimates for pending jobs and shows
  `StartTime` in the table. Unavailable start times are `null` in JSON.
- In `job list --json`, `ok` is `true` for completed jobs, `false` for
  terminal failures, and `null` for jobs still in progress. `job status --json`
  includes each accounting row, including job steps such as `524614.batch`.
  Accounting rows include integer `exit_code` and `signal` fields when Slurm
  reports a return code and signal.
- `job submit -W/--wait` blocks until the job finishes and `rcc` exits with the
  job's exit code — closing the submit→monitor loop in one command. `--dependency`
  passes `--dependency=<TYPE:JOBID>` straight to sbatch for chaining (e.g.
  `afterok:524614`).
- `job tail` reads `slurm-<JOBID>.out` inside `remote_dir` by default. Pass
  `--file NAME` for jobs that set a custom `--output`.
- `job wait` polls `squeue` until the job leaves the queue, then classifies the
  final state via `sacct`: exits `0` on `COMPLETED`, the job's exit code on
  failure, `124` on timeout. `--on STATE` returns early (e.g. once `RUNNING`).
  Unlike `sbatch --wait`, it surfaces the job's outcome without minutes-long
  silence.

For one-off Slurm commands that aren't wrapped, use the shell-string mode of
`rcc run`, which also sidesteps the quoting problem:

```bash
rcc run -s 'sacct -j 524614 --format=JobID,State,Elapsed,ExitCode,Reason'
```

## Roadmap / not yet implemented

Tracked in the issue tracker; the four originally-deferred asks from issues #2/#3
are now implemented (run tee, detached tmux runs, `job wait`, `status --json`).
What remains is the Slurm-specific parsing work, which needs a live scheduler
to validate:
- **Rank-aware Slurm logs** (`job tail --rank N`, distributing across the job's
  node list) — issue #2 P2. Needs a multi-node job to validate the node/rank
  mapping; deferred until testable against a live scheduler.

`rcc config --json`, `rcc status --json`, **and** `rcc job list/status --json`
*are* available.

## Releasing (maintainers)

Releases are published to PyPI via [**trusted publishing**](https://docs.pypi.org/trusted-publishers/) (OIDC): no API tokens are stored anywhere. A GitHub Actions workflow (`.github/workflows/release.yml`) builds the sdist + wheel and publishes under the `pypi` environment, where PyPI trusts it by repository + workflow filename + environment.

### One-time setup (PyPI web UI)

Only the PyPI project owner can do this. Go to
<https://pypi.org/manage/project/remote-cluster-controller/settings/publishing/>,
under **Add a publisher → GitHub**, and register:

| Field | Value |
|---|---|
| Owner | `ResearchComputer` |
| Repository | `remote-cluster-controller` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

(The `Environment name` must match the `environment: pypi` line in the workflow.)

Until this is done, the first publish run will fail at the `Publish to PyPI` step
with a clear PyPI error about an unknown publisher.

Optional hardening: under the repo's **Settings → Environments → `pypi`**, add a
required reviewer so every publish needs a manual approval.

### Cutting a release

The git tag drives the release and **must match** the `version` in
`pyproject.toml` (a workflow step enforces this, so a mismatch fails fast
instead of publishing the wrong version):

```bash
# 1. bump version in pyproject.toml and src/rcc/__init__.py
# 2. commit, then:
git tag v0.3.2
git push origin v0.3.2
```

Pushing the tag triggers the workflow. Watch it:

```bash
gh run watch
```

There is also a `workflow_dispatch` trigger (Actions tab → Run workflow) for
re-runs; it publishes whatever `version` is in `pyproject.toml` on the selected
branch.

## Design

See the design docs for the full picture:

- [`docs/superpowers/specs/2026-04-23-rcc-design.md`](docs/superpowers/specs/2026-04-23-rcc-design.md) — original v1 design.
- [`docs/superpowers/specs/2026-06-28-rcc-issues-2-3.md`](docs/superpowers/specs/2026-06-28-rcc-issues-2-3.md) — issues #2 & #3: automation ergonomics, sync safety, lifecycle verbs, and a condensed usage tour.
