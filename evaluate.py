"""
evaluate.py -- Part 3: evaluation + error analysis for the underground-artist task.

Run:   python evaluate.py                      (reads study_dataset.csv)
       python evaluate.py --input my_file.csv

Needs the columns made by build_dataset*.py and label_with_gemini.py:
  correct_option, model_choice, option_A/B/C_type, favorite_genre, ...

Outputs (in the current folder):
  evaluation_results.txt   all numbers, ready to paste into the PDF
  error_distribution.csv   the error distribution table (pattern, count, % of errors)
  errors.csv               every wrong trial, with an empty 'manual_pattern' column for you to tag

Metric: accuracy (fraction of trials where the model's pick == the ground-truth artist), reported with a
95% Wilson confidence interval. Chance level for 3 options is 33.3%.
"""
import argparse
import math

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="study_dataset.csv")
args = parser.parse_args()

df = pd.read_csv(args.input, dtype={"model_choice": str})
df = df[df["model_choice"].isin(list("ABC"))].copy()
n = len(df)
if n == 0:
    raise SystemExit("No labeled trials found. Run label_with_gemini.py first.")

df["correct"] = df["model_choice"] == df["correct_option"]
df["chosen_type"] = [r[f"option_{r['model_choice']}_type"] for _, r in df.iterrows()]

lines = []


def out(s=""):
    print(s)
    lines.append(s)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------- 1. Overall ----------
k = int(df.correct.sum())
lo, hi = wilson(k, n)
out("=== 1. Overall accuracy ===")
out(f"Trials evaluated: {n}")
out(f"Correct: {k}   Accuracy: {k / n:.1%}   95% CI: [{lo:.1%}, {hi:.1%}]   (chance = 33.3%)")

# ---------- 2. Error distribution (by what the model chose instead) ----------
errs = df[~df.correct].copy()
out("\n=== 2. Error distribution (what the model chose when wrong) ===")
dist = errs.chosen_type.value_counts().rename_axis("pattern").reset_index(name="count")
dist["pct_of_errors"] = (dist["count"] / len(errs) * 100).round(1)
out(f"Total errors: {len(errs)}")
out(dist.to_string(index=False))
dist.to_csv("error_distribution.csv", index=False)

# ---------- 3. Accuracy by where the correct answer sits, and position bias ----------
out("\n=== 3. Accuracy by position of the correct answer ===")
by_pos = df.groupby("correct_option").correct.agg(["mean", "sum", "count"])
by_pos["mean"] = (by_pos["mean"] * 100).round(1)
out(by_pos.rename(columns={"mean": "acc_%", "sum": "correct", "count": "n"}).to_string())
out("\nHow often the model picked each position (should be ~33% each if unbiased):")
out((df.model_choice.value_counts(normalize=True).sort_index() * 100).round(1).to_string())

# ---------- 4. Accuracy by genre ----------
out("\n=== 4. Accuracy by favorite genre (genres with >= 5 trials) ===")
g = df.groupby("favorite_genre").correct.agg(["mean", "sum", "count"])
g = g[g["count"] >= 5].sort_values("mean")
g["mean"] = (g["mean"] * 100).round(1)
out(g.rename(columns={"mean": "acc_%", "sum": "correct", "count": "n"}).to_string())

# ---------- 5. Difficulty: does the disliked-genre decoy trap the model? ----------
out("\n=== 5. Accuracy by which decoy types appeared in the trial ===")
types = df[["option_A_type", "option_B_type", "option_C_type"]].apply(
    lambda r: " + ".join(sorted(t for t in r if t != "correct")), axis=1)
t = df.assign(decoys=types).groupby("decoys").correct.agg(["mean", "count"])
t["mean"] = (t["mean"] * 100).round(1)
out(t.rename(columns={"mean": "acc_%", "count": "n"}).to_string())

# ---------- 6. Error file for manual tagging ----------
cols = ["trial_id", "profile_text", "option_A", "option_B", "option_C", "correct_option",
        "model_choice", "chosen_type", "model_reason"]
e = errs[cols].copy()
e["manual_pattern"] = ""   # fill in by hand, e.g. "ignored dislike", "fooled by genre tag", "audio mismatch"
e.to_csv("errors.csv", index=False)
out(f"\nWrote errors.csv ({len(e)} rows) -- tag the 'manual_pattern' column for your own error table.")

with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nSaved evaluation_results.txt, error_distribution.csv, errors.csv")