"""
build_dataset_v2.py -- study dataset for "Find an underground artist someone would love".

Usage:   python build_dataset_v2.py dataset.csv
Input:   Kaggle "Spotify Tracks Dataset" (maharshipandya) CSV
Output:  study_dataset.csv (220 trials)

Changes vs v1 (found by reviewing v1's output):
  * "Underground" now means the artist's BEST track has popularity <= UNDERGROUND_MAX_TRACK_POP
    (v1 used the mean, which let famous artists like Diana Ross through).
  * Language/region/odd tags and classical-type genres removed (noisy or hard for classmates to judge).
  * Every option is a same-genre artist, except the disliked-genre decoy in ~25% of trials,
    so genre alone no longer eliminates an option.
  * Each decoy artist may appear at most MAX_USES times across the whole dataset.
  * Decoy types: same_genre_near_decoy (hard), same_genre_far_decoy (easy), disliked_genre_decoy.

Ground truth = the underground artist whose audio profile is closest to the average profile of
the listener's 3 favorites (a proxy for "would enjoy it"; state this limitation in your report).
"""
import sys
import random
from collections import Counter

import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
OUT = "study_dataset.csv"
SEED = 42
N_TRIALS = 220
FAMOUS_MIN_POP = 55             # mean popularity of artists the listener "knows and loves"
UNDERGROUND_MAX_TRACK_POP = 35  # best track of an underground artist must be <= this
MAX_USES = 2                    # max appearances of any single option artist
P_DISLIKED_DECOY = 0.25
NEAR_GAP = (0.4, 1.6)           # near decoy is this much farther (z-distance) than the correct artist
FEATS = ["danceability", "energy", "valence", "acousticness",
         "instrumentalness", "speechiness", "tempo"]
EXCLUDE = {
    # not real "listening moods" / kids / novelty
    "comedy", "children", "kids", "disney", "sleep", "study", "work-out", "party", "happy",
    "sad", "romance", "show-tunes", "new-age", "anime", "holidays",
    # languages / regions rather than genres (noisy tags)
    "british", "french", "german", "spanish", "swedish", "latino", "indian", "iranian",
    "malay", "turkish", "brazil", "world-music", "mandopop", "cantopop", "j-dance", "j-idol",
    "j-rock", "k-pop", "pagode", "samba", "sertanejo", "forro", "mpb", "tango",
    # classical-type tags are mislabeled a lot and hard for classmates to judge
    "classical", "opera", "piano", "guitar",
}

rng = random.Random(SEED)

# ---------- 1. Load and clean ----------
df = pd.read_csv(CSV)
df = df.dropna(subset=["artists", "track_name", "track_genre"])
df["artist"] = df["artists"].str.split(";").str[0].str.strip()
df = df.drop_duplicates(subset=["artist", "track_name"])
df = df[~df["track_genre"].isin(EXCLUDE)]

# ---------- 2. One row per artist ----------
top_track = (df.sort_values("popularity", ascending=False)
               .groupby("artist")["track_name"].first())
agg = df.groupby("artist").agg(
    popularity=("popularity", "mean"),
    max_pop=("popularity", "max"),
    n_tracks=("track_id", "count"),
    genre=("track_genre", lambda s: s.mode().iloc[0]),
    **{f: (f, "mean") for f in FEATS},
).reset_index()
agg["top_track"] = agg["artist"].map(top_track)

Z = (agg[FEATS] - agg[FEATS].mean()) / agg[FEATS].std()

famous = agg[(agg.popularity >= FAMOUS_MIN_POP) & (agg.n_tracks >= 3)]
under = agg[(agg.max_pop <= UNDERGROUND_MAX_TRACK_POP) & (agg.n_tracks >= 2)]

eligible = [g for g in sorted(agg.genre.unique())
            if (famous.genre == g).sum() >= 3 and (under.genre == g).sum() >= 10]
print(f"{len(eligible)} usable genres: {eligible}")
if len(eligible) < 2:
    sys.exit("Too few usable genres -- loosen FAMOUS_MIN_POP / UNDERGROUND_MAX_TRACK_POP.")

uses = Counter()


def describe(r):
    bits = ["high-energy" if r["energy"] >= 0.65 else
            "mellow" if r["energy"] < 0.35 else "medium-energy",
            "upbeat" if r["valence"] >= 0.65 else
            "dark and moody" if r["valence"] < 0.35 else "bittersweet"]
    if r["acousticness"] >= 0.6:
        bits.append("mostly acoustic")
    elif r["acousticness"] <= 0.15:
        bits.append("heavily produced/electric")
    if r["danceability"] >= 0.65:
        bits.append("danceable")
    if r["instrumentalness"] >= 0.5:
        bits.append("mostly instrumental")
    if r["speechiness"] >= 0.33:
        bits.append("lots of spoken/rapped vocals")
    if r["tempo"] >= 130:
        bits.append("fast tempo")
    elif r["tempo"] <= 95:
        bits.append("slow tempo")
    return ", ".join(bits)


def option_text(r):
    return (f'{r["artist"]} -- {r["genre"].replace("-", " ")}; {describe(r)}; '
            f'known for "{r["top_track"]}"')


def dist_to(pool, centroid):
    return pd.Series(np.linalg.norm(Z.loc[pool.index] - centroid, axis=1), index=pool.index)


def available(pool):
    return pool[[uses[a] < MAX_USES for a in pool["artist"]]]


# ---------- 3. Build trials ----------
rows, attempts = [], 0
while len(rows) < N_TRIALS and attempts < N_TRIALS * 40:
    g = eligible[attempts % len(eligible)]
    attempts += 1

    favs = famous[famous.genre == g].sample(3, random_state=rng.randrange(10**9))
    centroid = Z.loc[favs.index].mean()

    pool = available(under[under.genre == g]).copy()
    if len(pool) < 8:
        continue
    pool["dist"] = dist_to(pool, centroid)
    pool = pool.sort_values("dist").reset_index(drop=True)

    correct = pool.iloc[rng.randrange(0, 3)]
    near = pool[(pool.dist >= correct.dist + NEAR_GAP[0]) & (pool.dist <= correct.dist + NEAR_GAP[1])]
    far = pool.iloc[int(len(pool) * 0.75):]
    far = far[far.dist >= correct.dist + 1.5]
    if near.empty or far.empty:
        continue
    near_decoy = near.iloc[rng.randrange(len(near))]

    dislike = rng.choice([x for x in eligible if x != g])
    if rng.random() < P_DISLIKED_DECOY:
        pool_b = available(under[under.genre == dislike])
        if pool_b.empty:
            continue
        other = pool_b.iloc[rng.randrange(len(pool_b))]
        other_type = "disliked_genre_decoy"
    else:
        other = far.iloc[rng.randrange(len(far))]
        other_type = "same_genre_far_decoy"

    correct_pos = "ABC"[len(rows) % 3]
    decoys = [(near_decoy, "same_genre_near_decoy"), (other, other_type)]
    rng.shuffle(decoys)
    slots, it = {}, iter(decoys)
    for p in "ABC":
        slots[p] = (correct, "correct") if p == correct_pos else next(it)

    row = {
        "trial_id": len(rows) + 1,
        "favorite_genre": g,
        "favorite_artists": "; ".join(favs.artist),
        "disliked_genre": dislike,
        "profile_text": (f"Listener's favorite artists: {', '.join(favs.artist)} "
                         f"(all {g.replace('-', ' ')}). "
                         f"They dislike {dislike.replace('-', ' ')} music."),
    }
    for p in "ABC":
        r, kind = slots[p]
        row[f"option_{p}"] = option_text(r)
        row[f"option_{p}_type"] = kind
        uses[r["artist"]] += 1
    row["correct_option"] = correct_pos
    row["correct_artist"] = correct["artist"]
    row["model_choice"] = ""
    row["model_reason"] = ""
    row["model_correct"] = ""
    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

# ---------- 4. Sanity report ----------
print(f"\nWrote {len(out)} trials to {OUT}")
if len(out) < 200:
    print("WARNING: fewer than 200 trials -- loosen thresholds (MAX_USES, UNDERGROUND_MAX_TRACK_POP, NEAR_GAP).")
print("\nCorrect-answer position:\n", out.correct_option.value_counts().to_string())
print("\nGenres used:", out.favorite_genre.nunique())
print("\nDecoy types:\n", pd.concat([out[f"option_{p}_type"] for p in "ABC"]).value_counts().to_string())
names = pd.concat([out[f"option_{p}"] for p in "ABC"]).str.split(" -- ").str[0]
print(f"\nDistinct option artists: {names.nunique()} of {len(names)} (max uses {names.value_counts().max()})")