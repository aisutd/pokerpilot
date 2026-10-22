from pokerpilot_model.run_reports import write_run_report


def test_each_run_writes_a_new_markdown_file(tmp_path):
    results = {
        "split": "test", "rows": 3,
        "model_accuracy": 2 / 3, "model_macro_f1": 0.5,
        "baseline_accuracy": 1 / 3, "baseline_macro_f1": 0.2,
        "action_report": {
            "fold": {"support": 2, "precision": 1.0, "recall": 0.5, "f1-score": 2 / 3},
            "macro avg": {"support": 3, "precision": 0.5, "recall": 0.5, "f1-score": 0.5},
        },
        "action_counts": {"fold": 2, "call": 1},
    }
    first = write_run_report(tmp_path, "test", results)
    second = write_run_report(tmp_path, "test", results)

    assert first != second
    assert first.exists() and second.exists()
    assert first.suffix == ".md"
    assert "Model and baseline" in first.read_text()
