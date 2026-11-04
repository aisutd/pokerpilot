"""Write one comparison-friendly Markdown file per CLI evaluation run."""

from datetime import datetime, timezone
from pathlib import Path


def _metric_rows(report: dict) -> list[str]:
    return [
        f"| `{label}` | {int(values['support']):,} | {values['precision']:.4f} "
        f"| {values['recall']:.4f} | {values['f1-score']:.4f} |"
        for label, values in report.items()
        if isinstance(values, dict) and "f1-score" in values
        and label not in {"macro avg", "weighted avg", "accuracy"}
    ]


def render_training(results: dict) -> str:
    lines = [
        "# PokerPilot Training and Test Run",
        "",
        f"- Random seed: `{results['seed']}`",
        f"- Selected boosting rounds: `{results['selected_estimators']}`",
        f"- Selected maximum tree depth: `{results['selected_max_depth']}`",
        f"- Class-weight power: `{results['selected_class_weight_power']}`",
        f"- Features: {len(results['features'])}",
        "",
        "## Split metrics",
        "",
        "| Split | Decisions | Games | Accuracy | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("validation", "test"):
        scores = results[split]
        rows = results["rows"]
        games = results["unique_games"]
        lines.append(
            f"| {split.title()} | {rows['val' if split == 'validation' else 'test']:,} "
            f"| {games['val' if split == 'validation' else 'test']:,} "
            f"| {scores['accuracy']:.4%} | {scores['macro_f1']:.4f} |"
        )
    lines += ["", "## Test metrics by action", "",
              "| Action | Support | Precision | Recall | F1 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    lines.extend(_metric_rows(results["test"]["report"]))
    lines += ["", "## Test comparison", "",
              "| Strategy | Accuracy | Macro F1 |",
              "| --- | ---: | ---: |",
              f"| XGBoost | {results['test']['accuracy']:.4%} | {results['test']['macro_f1']:.4f} |",
              f"| Equity and pot-odds rule | {results['baselines']['equity_pot_odds']['accuracy']:.4%} "
              f"| {results['baselines']['equity_pot_odds']['macro_f1']:.4f} |",
              f"| Always predict `{results['majority_action']}` | {results['baselines']['majority']['accuracy']:.4%} "
              f"| {results['baselines']['majority']['macro_f1']:.4f} |",
              "", "## Run details", "",
              f"- Labels: {', '.join(f'`{label}`' for label in results['labels'])}",
              "- Test split was not used for model selection.",
              "- These scores measure agreement with recorded actions, not poker profit.", ""]
    return "\n".join(lines)


def render_replay(results: dict) -> str:
    lines = [
        f"# PokerPilot {results['split'].title()} Replay",
        "",
        f"- Decisions: {results['rows']:,}",
        "",
        "## Model and baseline",
        "",
        "| Strategy | Accuracy | Macro F1 |",
        "| --- | ---: | ---: |",
        f"| XGBoost model | {results['model_accuracy']:.4%} | {results['model_macro_f1']:.4f} |",
        f"| Equity and pot-odds rule | {results['baseline_accuracy']:.4%} | {results['baseline_macro_f1']:.4f} |",
        "",
        "## Model metrics by action",
        "",
        "| Action | Support | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(_metric_rows(results["action_report"]))
    lines += ["", "## Predicted action counts", "",
              "| Action | Count |", "| --- | ---: |"]
    lines.extend(f"| `{action}` | {count:,} |"
                 for action, count in sorted(results["action_counts"].items()))
    lines += ["", "This replay scores predictions against recorded actions; it is not a counterfactual EV estimate.", ""]
    return "\n".join(lines)


def write_run_report(report_dir: Path, kind: str, results: dict) -> Path:
    """Create a unique UTC-stamped Markdown report and return its path."""
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = report_dir / f"{kind}_{timestamp}.md"
    content = render_training(results) if kind == "training" else render_replay(results)
    with path.open("x", encoding="utf-8") as report:
        report.write(content)
    return path
