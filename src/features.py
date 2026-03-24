"""
Feature Engineering Pipeline
=============================
Compute team-level stats per season and matchup-level difference features.
"""

import numpy as np
import pandas as pd
from configs.config import MENS_TEAM_ID_MIN, MENS_TEAM_ID_MAX


def compute_team_season_stats(regular_detailed_df):
    """
    Compute per-team, per-season aggregate statistics from detailed results.

    Returns DataFrame with columns: Season, TeamID, + all computed stats.
    """
    df = regular_detailed_df.copy()

    # Build stats from both winning and losing perspectives
    win_stats = df.rename(columns=lambda c: c.replace("W", "T_").replace("L", "O_") if c[0] in ("W", "L") and len(c) > 1 else c)
    loss_stats = df.rename(columns=lambda c: c.replace("L", "T_").replace("W", "O_") if c[0] in ("W", "L") and len(c) > 1 else c)

    # Standardize - mark wins/losses
    win_stats["Win"] = 1
    loss_stats["Win"] = 0

    # Rename team ID columns
    win_stats = win_stats.rename(columns={"T_TeamID": "TeamID", "O_TeamID": "OppID",
                                           "T_Score": "Score", "O_Score": "OppScore"})
    loss_stats = loss_stats.rename(columns={"T_TeamID": "TeamID", "O_TeamID": "OppID",
                                             "T_Score": "Score", "O_Score": "OppScore"})

    all_games = pd.concat([win_stats, loss_stats], ignore_index=True)

    # Compute per-team per-season aggregates
    agg_funcs = {
        "Win": ["sum", "count"],
        "Score": "mean",
        "OppScore": "mean",
    }

    # Add detailed stat columns if they exist
    detail_cols = ["T_FGM", "T_FGA", "T_FGM3", "T_FGA3", "T_FTM", "T_FTA",
                   "T_OR", "T_DR", "T_Ast", "T_TO", "T_Stl", "T_Blk", "T_PF",
                   "O_FGM", "O_FGA", "O_FGM3", "O_FGA3", "O_FTM", "O_FTA",
                   "O_OR", "O_DR", "O_Ast", "O_TO", "O_Stl", "O_Blk", "O_PF"]

    for col in detail_cols:
        if col in all_games.columns:
            agg_funcs[col] = "mean"

    grouped = all_games.groupby(["Season", "TeamID"]).agg(agg_funcs)
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]
    grouped = grouped.rename(columns={"Win_sum": "Wins", "Win_count": "Games"})
    grouped["WinPct"] = grouped["Wins"] / grouped["Games"]
    grouped["PointDiff"] = grouped["Score_mean"] - grouped["OppScore_mean"]

    # Four Factors
    if "T_FGA_mean" in grouped.columns:
        grouped["eFG_pct"] = (grouped["T_FGM_mean"] + 0.5 * grouped["T_FGM3_mean"]) / grouped["T_FGA_mean"]
        grouped["TO_pct"] = grouped["T_TO_mean"] / (grouped["T_FGA_mean"] + 0.44 * grouped["T_FTA_mean"] + grouped["T_TO_mean"])
        grouped["ORB_pct"] = grouped["T_OR_mean"] / (grouped["T_OR_mean"] + grouped["O_DR_mean"])
        grouped["FT_rate"] = grouped["T_FTM_mean"] / grouped["T_FGA_mean"]

        # Opponent Four Factors
        grouped["Opp_eFG_pct"] = (grouped["O_FGM_mean"] + 0.5 * grouped["O_FGM3_mean"]) / grouped["O_FGA_mean"]
        grouped["Opp_TO_pct"] = grouped["O_TO_mean"] / (grouped["O_FGA_mean"] + 0.44 * grouped["O_FTA_mean"] + grouped["O_TO_mean"])

        # Efficiency (points per possession estimate)
        possessions = grouped["T_FGA_mean"] - grouped["T_OR_mean"] + grouped["T_TO_mean"] + 0.44 * grouped["T_FTA_mean"]
        grouped["OffRating"] = grouped["Score_mean"] / possessions * 100
        opp_poss = grouped["O_FGA_mean"] - grouped["O_OR_mean"] + grouped["O_TO_mean"] + 0.44 * grouped["O_FTA_mean"]
        grouped["DefRating"] = grouped["OppScore_mean"] / opp_poss * 100
        grouped["NetRating"] = grouped["OffRating"] - grouped["DefRating"]
        grouped["Pace"] = (possessions + opp_poss) / 2

    grouped = grouped.reset_index()
    return grouped


def compute_seed_features(seeds_df):
    """Extract numeric seed from seed string (e.g., 'W01' -> 1)."""
    df = seeds_df.copy()
    df["SeedNum"] = df["Seed"].str[1:3].astype(int)
    return df[["Season", "TeamID", "SeedNum"]]


def build_matchup_features(team_stats, seeds, elo_ratings, team_a_id, team_b_id, season):
    """
    Build feature vector for a single matchup (TeamA vs TeamB).
    TeamA should have the lower TeamID.
    Returns dict of features.
    """
    # Get stats for both teams
    a_stats = team_stats[(team_stats["Season"] == season) & (team_stats["TeamID"] == team_a_id)]
    b_stats = team_stats[(team_stats["Season"] == season) & (team_stats["TeamID"] == team_b_id)]

    if len(a_stats) == 0 or len(b_stats) == 0:
        return None

    a = a_stats.iloc[0]
    b = b_stats.iloc[0]

    features = {}

    # Seed features
    a_seed = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team_a_id)]
    b_seed = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team_b_id)]
    if len(a_seed) > 0 and len(b_seed) > 0:
        features["seed_diff"] = a_seed.iloc[0]["SeedNum"] - b_seed.iloc[0]["SeedNum"]
    else:
        features["seed_diff"] = 0

    # Elo features
    if elo_ratings is not None:
        a_elo = elo_ratings[(elo_ratings["Season"] == season) & (elo_ratings["TeamID"] == team_a_id)]
        b_elo = elo_ratings[(elo_ratings["Season"] == season) & (elo_ratings["TeamID"] == team_b_id)]
        if len(a_elo) > 0 and len(b_elo) > 0:
            features["elo_diff"] = a_elo.iloc[0]["EloRating"] - b_elo.iloc[0]["EloRating"]
        else:
            features["elo_diff"] = 0

    # Difference features for all numeric stats
    diff_cols = ["WinPct", "PointDiff", "Score_mean", "OppScore_mean"]
    optional_cols = ["eFG_pct", "TO_pct", "ORB_pct", "FT_rate",
                     "OffRating", "DefRating", "NetRating", "Pace",
                     "Opp_eFG_pct", "Opp_TO_pct"]

    for col in diff_cols + [c for c in optional_cols if c in a.index]:
        features[f"{col}_diff"] = a[col] - b[col]

    # Interaction features
    if "elo_diff" in features:
        features["seed_x_elo_diff"] = features["seed_diff"] * features["elo_diff"]

    return features


def build_training_data(team_stats, seeds, elo_ratings, tourney_results_df):
    """
    Build full training dataset from historical tournament results.
    Each row = one tournament game with features and outcome.
    """
    rows = []

    for _, game in tourney_results_df.iterrows():
        season = game["Season"]
        w_id = game["WTeamID"]
        l_id = game["LTeamID"]

        # Ensure lower ID is team_a
        team_a = min(w_id, l_id)
        team_b = max(w_id, l_id)
        target = 1 if team_a == w_id else 0  # Did lower ID win?

        features = build_matchup_features(team_stats, seeds, elo_ratings, team_a, team_b, season)
        if features is not None:
            features["Season"] = season
            features["TeamA"] = team_a
            features["TeamB"] = team_b
            features["Target"] = target
            rows.append(features)

    return pd.DataFrame(rows)
