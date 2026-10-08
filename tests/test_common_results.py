import json

import pytest

from common.results import append_result, summarize


def test_append_result_writes_one_json_line_per_call(tmp_path):
    append_result("s99", {"seed": 0, "dev_loss": 2.5}, results_dir=tmp_path)
    path = append_result("s99", {"seed": 1, "dev_loss": 2.4}, results_dir=tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"step": "s99", "seed": 0, "dev_loss": 2.5},
        {"step": "s99", "seed": 1, "dev_loss": 2.4},
    ]


def test_summarize_groups_runs_and_reports_mean_and_std(tmp_path):
    for seed, loss in enumerate([2.0, 2.2, 2.4]):
        append_result("s99", {"seed": seed, "lr": 0.1, "dev_loss": loss}, results_dir=tmp_path)
    append_result("s99", {"seed": 0, "lr": 0.5, "dev_loss": 3.0}, results_dir=tmp_path)
    by_lr = {row["lr"]: row for row in summarize("s99", "lr", "dev_loss", results_dir=tmp_path)}
    assert by_lr[0.1]["n"] == 3
    assert by_lr[0.1]["mean"] == pytest.approx(2.2)
    assert by_lr[0.1]["std"] == pytest.approx(0.2)  # écart-type d'échantillon (n - 1)
    assert by_lr[0.5]["n"] == 1
    assert by_lr[0.5]["std"] is None  # un seul run : pas d'écart-type
