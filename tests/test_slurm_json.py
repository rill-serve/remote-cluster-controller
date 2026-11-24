"""Tests for the structured (JSON) Slurm listing/status — issue #2 P2.

The parser is the fragile bit the issues complain about, so it is tested
exhaustively against realistic squeue/sacct fixtures (header+rows for sacct,
known-column rows for squeue), plus the enrichment (exit_code/signal/ok) and
the SSH-backed ops (mocked).
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from rcc import slurm
from rcc.config import Profile
from rcc.errors import RemoteError

PROFILE = Profile(host="h", remote_dir="/r")
TABLE_FIELDS = ["JobID", "Name", "Partition", "State", "Time", "NodeList"]


# ------------------------------- builders ----------------------------------- #


def test_squeue_script_uses_explicit_format_no_header():
    s = slurm.squeue_script()
    assert "squeue -h " in s
    assert "\x1f".join("%i %j %P %T %M %N %r %l %S %E %a %V %D".split()) in s
    assert "id -un" in s  # current-user expansion
    assert "--start" not in s


def test_squeue_start_script_uses_job_id_and_start_time():
    s = slurm.squeue_script(start=True)
    assert "--start --states=PENDING " in s
    assert "%i\x1f%S" in s


def test_sacct_json_argv_uses_parsable_and_format():
    argv = slurm.sacct_json_argv("524614")
    assert argv[0] == "sacct"
    assert "-j" in argv and "524614" in argv
    assert "--parsable" in argv
    assert any("--format=" in a for a in argv)


# ------------------------------- parse_rows --------------------------------- #


def test_parse_rows_fixed_fields_squeue():
    out = "12345|train|gpu|RUNNING|5:00|node[01-02]\n12346|eval|cpu|PENDING|0:00|\n"
    rows = slurm.parse_rows(out, TABLE_FIELDS)
    assert len(rows) == 2
    assert rows[0]["JobID"] == "12345"
    assert rows[0]["State"] == "RUNNING"
    assert rows[0]["NodeList"] == "node[01-02]"
    assert rows[1]["State"] == "PENDING"
    assert rows[1]["NodeList"] == ""  # trailing empty field preserved
    assert rows[1]["Partition"] == "cpu"


def test_parse_rows_header_driven_sacct():
    out = (
        "JobID|JobName|Partition|State|Elapsed|ExitCode|Reason|Start|End\n"
        "524614|train|gpu|COMPLETED|01:00:00|0:0|None|2026-06-28T10:00:00|2026-06-28T11:00:00\n"
        "524614.batch|batch|gpu|COMPLETED|01:00:00|0:0|None|2026-06-28T10:00:00|2026-06-28T11:00:00\n"
    )
    rows = slurm.parse_rows(out)
    assert len(rows) == 2
    assert rows[0]["JobID"] == "524614"
    assert rows[0]["JobName"] == "train"
    assert rows[1]["JobID"] == "524614.batch"  # step distinguished by JobID


def test_parse_rows_empty_input():
    assert slurm.parse_rows("", TABLE_FIELDS) == []
    assert slurm.parse_rows("") == []


def test_parse_rows_header_only_no_data():
    assert (
        slurm.parse_rows(
            "JobID|State\n",
        )
        == []
    )


def test_parse_rows_skips_blank_lines_and_strips():
    out = "\n  12345 | train | gpu | RUNNING \n\n"
    rows = slurm.parse_rows(out, TABLE_FIELDS)
    assert rows == [
        {
            "JobID": "12345",
            "Name": "train",
            "Partition": "gpu",
            "State": "RUNNING",
            "Time": "",
            "NodeList": "",
        }
    ]


def test_parse_rows_ragged_row_is_padded():
    # A row missing trailing fields must not misalign (padded to header length).
    rows = slurm.parse_rows("1|a\n", TABLE_FIELDS)
    assert rows[0]["JobID"] == "1"
    assert rows[0]["Name"] == "a"
    assert rows[0]["NodeList"] == ""
    assert len(rows[0]) == len(TABLE_FIELDS)


def test_parse_rows_handles_nodeless_placeholder():
    # squeue emits the literal text "None" when there is no node assigned yet.
    rows = slurm.parse_rows("1|a|gpu|PENDING|0:00|None\n", TABLE_FIELDS)
    assert rows[0]["NodeList"] == "None"  # preserved verbatim, not coerced


# ------------------------------- enrichment --------------------------------- #


def test_enrich_adds_exit_code_and_signal():
    row = slurm._enrich({"JobID": "1", "State": "FAILED", "ExitCode": "2:9"})
    assert row["exit_code"] == 2
    assert row["signal"] == 9
    assert row["ok"] is False
    # original fields preserved
    assert row["JobID"] == "1" and row["ExitCode"] == "2:9"


def test_enrich_completed_ok_and_zero_exit():
    row = slurm._enrich({"State": "COMPLETED", "ExitCode": "0:0"})
    assert row["ok"] is True
    assert row["exit_code"] == 0 and row["signal"] == 0


def test_enrich_signal_killed():
    # ExitCode 0:15 -> killed by SIGTERM; ok False.
    row = slurm._enrich({"State": "CANCELLED", "ExitCode": "0:15"})
    assert row["ok"] is False
    assert row["exit_code"] == 0 and row["signal"] == 15


def test_enrich_no_exitcode_field():
    row = slurm._enrich({"State": "RUNNING"})
    assert "exit_code" not in row
    assert "signal" not in row
    assert row["ok"] is False


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("COMPLETED", True),
        ("FAILED", False),
        ("CANCELLED", False),
        ("TIMEOUT", False),
        ("OUT_OF_MEMORY", False),
        ("NODE_FAIL", False),
        ("PREEMPTED", False),
        ("BOOT_FAIL", False),
        ("DEADLINE", False),
        ("RUNNING", None),
        ("PENDING", None),
        ("COMPLETING", None),
        ("REQUEUED", None),
        ("UNKNOWN", None),
    ],
)
def test_queue_ok_reports_only_terminal_outcomes(state, expected):
    assert slurm._queue_ok(state) is expected


def test_enrich_garbage_exitcode_yields_none():
    row = slurm._enrich({"State": "FAILED", "ExitCode": "weird"})
    assert row["exit_code"] is None
    assert row["signal"] is None


def test_enrich_no_state_field():
    row = slurm._enrich({"JobID": "1"})
    assert "ok" not in row


# ------------------------------- SSH-backed ops ----------------------------- #


def _cap(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def _squeue_output(*rows: str) -> str:
    return "\n".join(row.replace("|", "\x1f") for row in rows)


def test_list_jobs_json_parses_squeue_output():
    squeue_out = _squeue_output(
        "12345|train|gpu|RUNNING|5:00|node01|None|1:30|2026-07-03T17:00:00||a|2026-07-03T16:00:00|1",
        "12346|eval|cpu|PENDING|0:00|None|Dependency|1:30|N/A|afterok:12345|a|2026-07-03T16:00:00|2",
    )
    with (
        patch("rcc.slurm.run_remote_capture", return_value=_cap(squeue_out)),
        patch("rcc.slurm.ensure_slurm"),
    ):
        rows = slurm.list_jobs_json(PROFILE)
    assert [r["JobID"] for r in rows] == ["12345", "12346"]
    assert rows[0]["ok"] is None
    assert "exit_code" not in rows[0]  # squeue output has no ExitCode field
    assert rows[1] == {
        "JobID": "12346",
        "Name": "eval",
        "Partition": "cpu",
        "State": "PENDING",
        "Time": "0:00",
        "NodeList": "None",
        "Reason": "Dependency",
        "TimeLimit": "1:30",
        "StartTime": None,
        "Dependency": "afterok:12345",
        "Account": "a",
        "SubmitTime": "2026-07-03T16:00:00",
        "NumNodes": "2",
        "ok": None,
    }


def test_list_jobs_json_merges_estimates_without_removing_active_jobs():
    queue = _squeue_output(
        "100|train|gpu|RUNNING|5:00|node01|None|1:30|17:00||a|16:00|1",
        "101_4|eval|cpu|PENDING|0:00||Priority|1:30|Unknown||a|16:00|1",
        "102|eval|cpu|PENDING|0:00||Priority|1:30|17:30||a|16:00|1",
        "103|eval|cpu|PENDING|0:00||Priority|1:30|None||a|16:00|1",
    )
    estimates = _squeue_output("104|20:00", "102|N/A", "101_4|18:00")
    with (
        patch("rcc.slurm.run_remote_capture", side_effect=[_cap(queue), _cap(estimates)]) as cap,
        patch("rcc.slurm.ensure_slurm"),
    ):
        rows = slurm.list_jobs_json(PROFILE, start=True)
    assert [(row["JobID"], row["StartTime"]) for row in rows] == [
        ("100", "17:00"), ("101_4", "18:00"), ("102", None), ("103", None),
    ]
    assert "--start" in cap.call_args.kwargs["script"]


def test_list_jobs_json_surfaces_squeue_failure(capsys):
    result = _cap("", 1, "squeue: error: Communication connection failure\n")
    with (
        patch("rcc.slurm.run_remote_capture", return_value=result),
        patch("rcc.slurm.ensure_slurm"),
        pytest.raises(RemoteError, match="squeue exited non-zero") as error,
    ):
        slurm.list_jobs_json(PROFILE)
    assert error.value.exit_code == 1
    assert "Communication connection failure" in capsys.readouterr().err


def test_list_jobs_json_empty_when_no_jobs():
    with (
        patch("rcc.slurm.run_remote_capture", return_value=_cap("")),
        patch("rcc.slurm.ensure_slurm"),
    ):
        assert slurm.list_jobs_json(PROFILE) == []


def test_status_json_parses_sacct_with_steps():
    sacct_out = (
        "JobID|JobName|Partition|State|Elapsed|ExitCode|Reason|Start|End\n"
        "524614|train|gpu|COMPLETED|01:00:00|0:0|None|10:00|11:00\n"
        "524614.batch|batch|gpu|COMPLETED|01:00:00|0:0|None|10:00|11:00\n"
        "524614.0|python|gpu|COMPLETED|00:59:00|0:0|None|10:00|10:59\n"
    )
    with (
        patch("rcc.slurm.run_remote_capture", return_value=_cap(sacct_out)),
        patch("rcc.slurm.ensure_slurm"),
    ):
        rows = slurm.status_json(PROFILE, "524614")
    assert len(rows) == 3
    assert rows[0]["JobID"] == "524614"
    assert rows[1]["JobID"] == "524614.batch"
    assert all(r["ok"] for r in rows)
    assert rows[0]["exit_code"] == 0


def test_status_json_failed_job_exit_code():
    sacct_out = (
        "JobID|JobName|Partition|State|Elapsed|ExitCode|Reason|Start|End\n"
        "9|boom|gpu|FAILED|00:01:00|3:0|OutOfMemory|10:00|10:01\n"
    )
    with (
        patch("rcc.slurm.run_remote_capture", return_value=_cap(sacct_out)),
        patch("rcc.slurm.ensure_slurm"),
    ):
        rows = slurm.status_json(PROFILE, "9")
    assert rows[0]["State"] == "FAILED"
    assert rows[0]["exit_code"] == 3
    assert rows[0]["ok"] is False
    assert rows[0]["Reason"] == "OutOfMemory"
