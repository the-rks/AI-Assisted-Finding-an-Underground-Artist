"""
label_with_gemini.py -- Part 2: have an AI model pick the artist for every trial.

Setup (once):
    pip install google-genai pandas
    Get a free key at https://aistudio.google.com/apikey  (no credit card)
    PowerShell:  $env:GEMINI_API_KEY="your-key"        (Mac/Linux: export GEMINI_API_KEY="your-key")

Run:
    python label_with_gemini.py --list-models                 # model names your key can use
    python label_with_gemini.py --limit 5 --batch 5           # small test (screenshot this)
    python label_with_gemini.py --model MODEL_NAME --batch 10 # label everything, 10 trials per API call

--batch N packs N trials into ONE request, so 220 trials need only ~22 calls. Use this when your
free tier allows few requests per day (e.g. 20/day). Use --batch 1 for one trial per call.

Resumable: rows that already have a model_choice are skipped and the CSV is saved after every call.
"""
import argparse
import json
import os
import random
import sys
import time

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="study_dataset.csv")
parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
parser.add_argument("--limit", type=int, default=None, help="only label this many unlabeled trials")
parser.add_argument("--batch", type=int, default=10, help="trials per API call")
parser.add_argument("--sleep", type=float, default=8.0, help="seconds between calls")
parser.add_argument("--list-models", action="store_true")
parser.add_argument("--dry-run", action="store_true", help="no API; random picks, to test the pipeline")
args = parser.parse_args()

HEADER = """You are helping people discover music. For each trial below you get a listener profile and
three artists the listener has probably never heard of. For EACH trial, pick the ONE artist the
listener would most enjoy. Think about what the listener's favorite artists sound like, then compare
with each option. Treat every trial independently.

Respond with JSON only: a list with one object per trial, in this form:
[{"trial_id": <number>, "choice": "A" | "B" | "C", "reason": "<one sentence>"}]
"""

client = types = None
if not args.dry_run:
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY first (see the top of this file).")
    client = genai.Client(api_key=key)
    if args.list_models:
        for m in client.models.list():
            print(m.name)
        sys.exit(0)


def build_prompt(rows):
    parts = [HEADER]
    for _, r in rows.iterrows():
        parts.append(f"--- Trial {r['trial_id']} ---\n{r['profile_text']}\n"
                     f"Option A: {r['option_A']}\nOption B: {r['option_B']}\nOption C: {r['option_C']}\n")
    return "\n".join(parts)


def ask(rows):
    """Returns {trial_id: (choice, reason)} for the trials the model answered validly."""
    ids = set(int(x) for x in rows["trial_id"])
    if args.dry_run:
        return {i: (random.choice("ABC"), "dry run") for i in ids}
    prompt = build_prompt(rows)
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=args.model, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0, response_mime_type="application/json"))
            data = json.loads(resp.text)
            if isinstance(data, dict):  # tolerate {"results": [...]}
                data = next(v for v in data.values() if isinstance(v, list))
            out = {}
            for d in data:
                tid, ch = int(d["trial_id"]), str(d["choice"]).strip().upper()[:1]
                if tid in ids and ch in ("A", "B", "C"):
                    out[tid] = (ch, str(d.get("reason", "")).strip())
            if out:
                return out
            raise ValueError("no valid answers in response")
        except Exception as e:
            wait = 20 * (attempt + 1)
            print(f"   retry {attempt + 1} after error: {str(e)[:110]} (waiting {wait}s)")
            time.sleep(wait)
    return {}


df = pd.read_csv(args.input, dtype={"model_choice": str, "model_reason": str, "model_correct": str})
for c in ("model_choice", "model_reason", "model_correct"):
    df[c] = df[c].fillna("")
todo = df.index[df["model_choice"] == ""].tolist()
if args.limit:
    todo = todo[:args.limit]
n_calls = -(-len(todo) // args.batch)
print(f"{len(todo)} trials to label with {'(dry run)' if args.dry_run else args.model} "
      f"in {n_calls} calls of up to {args.batch} trials")

fails = 0
for start in range(0, len(todo), args.batch):
    idx = todo[start:start + args.batch]
    answers = ask(df.loc[idx])
    fails = fails + 1 if not answers else 0
    for i in idx:
        tid = int(df.loc[i, "trial_id"])
        if tid in answers:
            ch, reason = answers[tid]
            df.loc[i, "model_choice"] = ch
            df.loc[i, "model_reason"] = reason
            df.loc[i, "model_correct"] = str(ch == df.loc[i, "correct_option"])
    df.to_csv(args.input, index=False)
    print(f"[call {start // args.batch + 1}/{n_calls}] labeled {len(answers)}/{len(idx)} trials")
    if fails >= 2:
        print("\nStopping: 2 failed calls in a row (usually a quota limit). Progress is saved -- "
              "wait, or try a different --model, then run the same command again.")
        break
    if not args.dry_run and start + args.batch < len(todo):
        time.sleep(args.sleep)

done = df[df["model_choice"].isin(list("ABC"))].copy()
if len(done):
    acc = (done["model_choice"] == done["correct_option"]).mean()
    print(f"\nLabeled {len(done)}/{len(df)} trials. Accuracy so far: {acc:.1%} (chance = 33.3%)")