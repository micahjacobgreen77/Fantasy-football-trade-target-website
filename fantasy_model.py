"""Fantasy football trade analyzer — regression models for buy-low / sell-high signals.

Models (all OLS on season-to-date aggregates, min 3 games, PPR scoring):
  1. Opportunity model (position-specific): WR/TE PPR ~ targets + receiving air yards;
     RB PPR ~ targets + carries. Gap = actual - expected -> buy-low / sell-high.
  2. Red-zone xTD model: total TDs ~ RZ targets + RZ carries + goal-line targets + goal-line
     carries (from play-by-play, yardline_100 <= 20 / <= 10). Negative gap = TDs coming.
  3. Air-yards conversion: receiving yards ~ receiving air yards (WR/TE).
  4. Trend: last-2-week average PPR minus earlier-week average.

Injured players (Out/Doubtful/IR per the latest nflverse injury report) are excluded
from trade recommendations; Questionable players are flagged.

Data: nflverse (https://github.com/nflverse/nflverse-data), 2026 season.
Usage: pip install -r requirements.txt && python fantasy_model.py
"""
import os
import urllib.request
import gzip
import shutil
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

SOURCES = {
    "stats_player_week_2026.csv": "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2026.csv",
    "play_by_play_2026.csv.gz": "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.csv.gz",
    "injuries_2026.csv.gz": "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2026.csv.gz",
}


def fetch(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(SOURCES[name], path)
    return path


def ols(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return pred, y - pred, beta, r2


def main():
    # ---- weekly player stats -> season aggregates ----
    wk = pd.read_csv(fetch("stats_player_week_2026.csv"), low_memory=False)
    wk = wk[wk["season_type"] == "REG"].copy()
    print("weeks in data:", sorted(wk["week"].unique()))

    num_cols = ["targets", "receptions", "receiving_yards", "receiving_tds",
                "receiving_air_yards", "carries", "rushing_yards", "rushing_tds",
                "target_share", "wopr", "receiving_2pt_conversions",
                "rushing_2pt_conversions", "receiving_fumbles_lost",
                "rushing_fumbles_lost"]
    for c in num_cols:
        wk[c] = pd.to_numeric(wk[c], errors="coerce").fillna(0)

    g = wk.groupby(["player_id", "player_display_name", "position", "team"],
                   as_index=False).agg(
        games=("week", "nunique"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        rec_yards=("receiving_yards", "sum"),
        rec_tds=("receiving_tds", "sum"),
        air_yards=("receiving_air_yards", "sum"),
        carries=("carries", "sum"),
        rush_yards=("rushing_yards", "sum"),
        rush_tds=("rushing_tds", "sum"),
    )
    g["fum_lost"] = (wk.groupby("player_id")[["receiving_fumbles_lost",
                    "rushing_fumbles_lost"]].sum().sum(axis=1)
                     .reindex(g["player_id"]).fillna(0).values)
    g["two_pt"] = (wk.groupby("player_id")[["receiving_2pt_conversions",
                   "rushing_2pt_conversions"]].sum().sum(axis=1)
                   .reindex(g["player_id"]).fillna(0).values)
    # PPR scoring
    g["actual_ppr"] = (g["receptions"] + g["rec_yards"] / 10 + g["rec_tds"] * 6
                       + g["rush_yards"] / 10 + g["rush_tds"] * 6
                       + g["two_pt"] * 2 - g["fum_lost"] * 2)
    g["total_yards"] = g["rec_yards"] + g["rush_yards"]
    g["total_tds"] = g["rec_tds"] + g["rush_tds"]
    g["touches"] = g["targets"] + g["carries"]

    skill = g[g["position"].isin(["RB", "WR", "TE"]) & (g["games"] >= 3)].copy()
    print(f"skill players (3+ games): {len(skill)}")

    # ---- Model 1: position-specific opportunity ----
    skill["xP_opp"] = np.nan
    for pos, feats in {"WR": ["targets", "air_yards"],
                       "TE": ["targets", "air_yards"],
                       "RB": ["targets", "carries"]}.items():
        m = skill["position"] == pos
        pred, _, _, r2 = ols(skill.loc[m, "actual_ppr"].values,
                             skill.loc[m, feats].values)
        skill.loc[m, "xP_opp"] = pred
        print(f"Model 1 ({pos}: PPR ~ {' + '.join(feats)}): R^2 = {r2:.3f}")
    skill["gap_opp"] = skill["actual_ppr"] - skill["xP_opp"]

    # ---- Model 2: red-zone expected TDs (from play-by-play) ----
    pbp = pd.read_csv(fetch("play_by_play_2026.csv.gz"), low_memory=False)
    pbp["is_pass_att"] = (pbp["complete_pass"] == 1) | (pbp["incomplete_pass"] == 1)
    pbp["is_rush_att"] = pbp["rusher_player_name"].notna() & (pbp["play_type"] == "run")
    rz_stats = {}

    def add_rz(frame, idcol, yl, key):
        sub = frame[frame["yardline_100"] <= yl]
        for pid, n in sub.groupby(idcol).size().items():
            rz_stats.setdefault(pid, {})[key] = int(n)

    pt = pbp[pbp["is_pass_att"] & pbp["receiver_player_id"].notna()]
    rc = pbp[pbp["is_rush_att"] & pbp["rusher_player_id"].notna()]
    add_rz(pt, "receiver_player_id", 20, "rz_targets")
    add_rz(pt, "receiver_player_id", 10, "gl_targets")
    add_rz(rc, "rusher_player_id", 20, "rz_carries")
    add_rz(rc, "rusher_player_id", 10, "gl_carries")

    for c in ["rz_targets", "gl_targets", "rz_carries", "gl_carries"]:
        skill[c] = skill["player_id"].map(
            lambda pid: rz_stats.get(pid, {}).get(c, 0)).astype(float)
    skill["rz_opps"] = skill["rz_targets"] + skill["rz_carries"]
    pred, _, _, r2 = ols(skill["total_tds"].values,
                         skill[["rz_targets", "rz_carries",
                                "gl_targets", "gl_carries"]].values)
    skill["xTD"], skill["td_gap"] = pred, skill["total_tds"] - pred
    print(f"Model 2 (TDs ~ RZ targets + RZ carries + goal-line): R^2 = {r2:.3f}")

    # ---- Model 3: air-yard conversion (WR/TE) ----
    wrte = skill[skill["position"].isin(["WR", "TE"])]
    pred, _, _, r2 = ols(wrte["rec_yards"].values, wrte[["air_yards"]].values)
    skill.loc[wrte.index, "xRecYd"] = pred
    skill["gap_air"] = skill["rec_yards"] - skill["xRecYd"].fillna(skill["rec_yards"])
    print(f"Model 3 (WR/TE rec yards ~ air yards): R^2 = {r2:.3f}")

    # ---- Model 4: trend (last 2 weeks vs earlier) ----
    wk_skill = wk[wk["player_id"].isin(skill["player_id"])].copy()
    wk_skill["wk_ppr"] = (wk_skill["receptions"]
                         + pd.to_numeric(wk_skill["receiving_yards"],
                                         errors="coerce").fillna(0) / 10
                         + pd.to_numeric(wk_skill["receiving_tds"],
                                         errors="coerce").fillna(0) * 6
                         + pd.to_numeric(wk_skill["rushing_yards"],
                                         errors="coerce").fillna(0) / 10
                         + pd.to_numeric(wk_skill["rushing_tds"],
                                         errors="coerce").fillna(0) * 6)
    max_wk = wk_skill["week"].max()
    recent = wk_skill[wk_skill["week"] > max_wk - 2].groupby("player_id")["wk_ppr"].mean()
    early = wk_skill[wk_skill["week"] <= max_wk - 2].groupby("player_id")["wk_ppr"].mean()
    skill["trend"] = (skill["player_id"].map(recent)
                      - skill["player_id"].map(early)).fillna(0)
    print("Model 4 (trend): last-2-week avg PPR minus earlier-week avg")

    # ---- injuries: exclude Out/Doubtful/IR, flag Questionable ----
    inj = pd.read_csv(fetch("injuries_2026.csv.gz"), low_memory=False)
    latest = inj[inj["week"] == inj["week"].max()]
    status = dict(zip(latest["gsis_id"], latest["report_status"]))
    detail = dict(zip(latest["gsis_id"], latest["report_primary_injury"]))
    skill["inj_status"] = skill["player_id"].map(status).fillna("Healthy")
    skill["inj_detail"] = skill["player_id"].map(detail).fillna("")
    bad = skill["inj_status"].isin(["Out", "Doubtful", "IR"])
    print(f"excluding {bad.sum()} injured (Out/Doubtful/IR) players from recommendations")
    rec = skill[~bad].copy()
    rec["q_flag"] = rec["inj_status"] == "Questionable"

    # ---- outputs ----
    def show(title, frame, cols, n=12, ascending=True, key="gap_opp"):
        print(f"\n=== {title} ===")
        for _, r in frame.sort_values(key, ascending=ascending).head(n).iterrows():
            q = " [Q]" if r.get("q_flag") else ""
            vals = " | ".join(f"{c} {r[c]}" for c in cols)
            print(f"{r['player_display_name']} ({r['position']},{r['team']}){q} | {vals}")

    show("BUY LOW (scoring below opportunity)", rec[rec["touches"] >= 40],
         ["gap_opp"], key="gap_opp")
    show("SELL HIGH (scoring above opportunity)", rec[rec["touches"] >= 30],
         ["gap_opp"], n=10, ascending=False, key="gap_opp")
    show("DUE FOR TDs (red-zone chances, no scores)", rec[rec["rz_opps"] >= 6],
         ["td_gap"], key="td_gap")
    show("TD OVERPERFORMERS", rec[rec["rz_opps"] >= 4],
         ["td_gap"], n=8, ascending=False, key="td_gap")
    show("TRENDING UP", rec, ["trend"], key="trend", ascending=False)
    show("TRENDING DOWN", rec, ["trend"], key="trend")

    out = os.path.join(DATA_DIR, "model_output.csv")
    rec.to_csv(out, index=False)
    print(f"\nfull output -> {out}")


if __name__ == "__main__":
    main()
