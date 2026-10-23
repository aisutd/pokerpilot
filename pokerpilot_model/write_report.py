"""
PokerPilot: turn the training outputs in ./artifacts/ into one readable
Markdown report (artifacts/report.md).

train.py calls this automatically at the end. You can also run it on its
own to rebuild the report from the last training run (no retraining):

Run:  python3 write_report.py
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "artifacts"


def table(df):
    """Markdown table from a DataFrame (no extra libraries needed)."""
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        # Up to 3 decimals, no trailing zeros (769.0 -> 769, 0.050 -> 0.05)
        cells = [idx] + [f"{v:.3f}".rstrip("0").rstrip(".")
                         if isinstance(v, float) else v for v in row]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    return "\n".join(lines)


def write_report(out=OUT):
    m = json.loads((out / "metrics.json").read_text())
    t, a, b = m["test_winning_players"], m["test_all_players"], m["baselines"]

    headline = pd.DataFrame({
        "model (winners)": [t["accuracy"], t["macro_f1"], t["log_loss"]],
        "model (all players)": [a["accuracy"], a["macro_f1"], a["log_loss"]],
        "always fold": [b["always_fold_accuracy"], None, None],
        "fold/check rule": [b["fold_or_check_rule_accuracy"],
                            b["fold_or_check_rule_macro_f1"], None],
    }, index=pd.Index(["accuracy", "macro F1", "log loss"], name="metric"))
    headline = headline.astype(object).where(headline.notna(), "-")

    streets = pd.DataFrame(m["by_street"]).T
    streets.index.name = "street"
    streets["rows"] = streets["rows"].astype(int)

    imp = pd.read_csv(out / "feature_importance.csv", index_col=0).head(12)
    imp.index.name = "feature"

    weights = pd.read_csv(out / "player_weights.csv", index_col=0)
    params = pd.DataFrame({"value": m["best_params"]})
    params.index.name = "parameter"

    sections = [
        "# PokerPilot XGBoost Model Report",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}_",
        "## Cross-validation",
        m["cross_validation"]["method"],
        m["cross_validation"]["selection_note"],
        table(pd.DataFrame(m["cross_validation"]["folds"]).set_index("fold")),
        "## Action model: test set results",
        "Scored on held-out test decisions made by winning players "
        f"({', '.join(m['winning_players'])}). Classes: {', '.join(m['classes'])}.",
        table(headline),
        "## Per-class report and confusion matrix",
        "Confusion matrix rows = what the player did, columns = model's pick.",
        "```\n" + (out / "test_report.txt").read_text().rstrip() + "\n```",
        "## Accuracy by street",
        table(streets),
        "## Top 12 features (share of total gain)",
        table(imp),
        "## Sizing model (bet/raise size as fraction of pot)",
        f"- MAE on test: **{m['sizing_mae_pot_frac']:.3f}** pot\n"
        f"- Baseline (always median size): {m['sizing_baseline_mae']:.3f} pot",
        "## Best hyperparameters",
        f"Final model uses **{m['n_trees']}** trees.",
        table(params),
        "## Player skill weights",
        "Luck-adjusted EV bb/100 on train + val hands, and the training weight each player got.",
        table(weights),
    ]
    path = out / "report.md"
    path.write_text("\n\n".join(sections) + "\n")
    return path


if __name__ == "__main__":
    print(f"Wrote {write_report()}")
