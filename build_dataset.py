"""
build_dataset.py -- builds the study dataset for "Find an underground artist someone would love".

Usage:   python build_dataset.py dataset.csv
Input:   the Kaggle "Spotify Tracks Dataset" (maharshipandya) CSV
Output:  study_dataset.csv  (220 trials, one row per trial)

Ground truth (a proxy for "the listener would enjoy it"):
  The correct option is the underground artist, in the same genre as the listener's
  favorites, whose average audio profile is closest to the favorites' average profile.
Decoys:
  same_genre_decoy      -> same genre tag, but audio profile is far from the favorites
  disliked_genre_decoy  -> artist from the genre the listener says they dislike
  other_genre_decoy     -> artist from some unrelated genre
"""
import sys
import random

import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
OUT = "study_dataset.csv"
SEED = 42
N_TRIALS = 220
FAMOUS_MIN_POP = 55       # artists the listener "knows and loves"
UNDERGROUND_MAX_POP = 25  # the unfamiliar, underground artists
FEATS = ["danceability", "energy", "valence", "acousticness",
         "instrumentalness", "speechiness", "tempo"]
EXCLUDE = {"comedy", "children", "kids", "disney", "sleep", "study", "work-out",
           "party", "happy", "sad", "romance", "show-tunes", "new-age"}

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
    n_tracks=("track_id", "count"),
    genre=("track_genre", lambda s: s.mode().iloc[0]),
    **{f: (f, "mean") for f in FEATS},
).reset_index()
agg["top_track"] = agg["artist"].map(top_track)

# z-scored audio features so distances are comparable across features
Z = (agg[FEATS] - agg[FEATS].mean()) / agg[FEATS].std()

famous = agg[(agg.popularity >= FAMOUS_MIN_POP) & (agg.n_tracks >= 3)]
under = agg[(agg.popularity <= UNDERGROUND_MAX_POP) & (agg.n_tracks >= 2)]

eligible = [g for g in sorted(agg.genre.unique())
            if (famous.genre == g).sum() >= 3 and (under.genre == g).sum() >= 6]
print(f"{len(eligible)} usable genres: {eligible}")
if not eligible:
    sys.exit("No usable genres -- loosen FAMOUS_MIN_POP / UNDERGROUND_MAX_POP.")


def describe(r):
    """Turn raw audio features into a short human-readable vibe description."""
    bits = []
    bits.append("high-energy" if r["energy"] >= 0.65 else
                "mellow" if r["energy"] < 0.35 else "medium-energy")
    bits.append("upbeat" if r["valence"] >= 0.65 else
                "dark and moody" if r["valence"] < 0.35 else "bittersweet")
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
    g = r["genre"].replace("-", " ")
    return f'{r["artist"]} -- {g}; {describe(r)}; known for "{r["top_track"]}"'


def dist_to(pool, centroid):
    return pd.Series(np.linalg.norm(Z.loc[pool.index] - centroid, axis=1), index=pool.index)


# ---------- 3. Build trials ----------
rows, used_correct = [], set()
gi = 0
while len(rows) < N_TRIALS and gi < len(eligible) * 60:
    g = eligible[gi % len(eligible)]
    gi += 1

    favs = famous[famous.genre == g].sample(3, random_state=rng.randrange(10**9))
    centroid = Z.loc[favs.index].mean()

    pool = under[under.genre == g].copy()
    pool["dist"] = dist_to(pool, centroid)
    pool = pool.sort_values("dist")
    correct = pool.iloc[rng.randrange(0, 3)]            # one of the 3 closest
    decoy_a = pool.iloc[-1 - rng.randrange(0, 3)]       # one of the 3 farthest
    if correct["artist"] in used_correct or decoy_a["dist"] - correct["dist"] < 1.0:
        continue

    dislike = rng.choice([x for x in eligible if x != g])
    if rng.random() < 0.5:
        pool_b = under[under.genre == dislike].copy()
        b_type = "disliked_genre_decoy"
    else:
        pool_b = under[~under.genre.isin([g, dislike])].copy()
        b_type = "other_genre_decoy"
    pool_b["dist"] = dist_to(pool_b, centroid)
    pool_b = pool_b[pool_b.dist > pool_b.dist.median()]
    decoy_b = pool_b.iloc[rng.randrange(len(pool_b))]

    # place the correct answer at a balanced position, shuffle the decoys
    correct_pos = "ABC"[len(rows) % 3]
    decoys = [(decoy_a, "same_genre_decoy"), (decoy_b, b_type)]
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
    row["correct_option"] = correct_pos
    row["correct_artist"] = correct["artist"]
    # to be filled in during Part 2 / Part 3
    row["model_choice"] = ""
    row["model_reason"] = ""
    row["model_correct"] = ""
    rows.append(row)
    used_correct.add(correct["artist"])

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

# ---------- 4. Sanity report ----------
print(f"\nWrote {len(out)} trials to {OUT}")
if len(out) < 200:
    print("WARNING: fewer than 200 trials -- loosen the popularity thresholds.")
print("\nCorrect-answer position:\n", out.correct_option.value_counts())
print("\nTop genres:\n", out.favorite_genre.value_counts().head(10))
print("\nDecoy types:\n", pd.concat([out[f"option_{p}_type"] for p in "ABC"]).value_counts())