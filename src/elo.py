"""
Elo Rating System
=================
Custom Elo ratings for NCAA basketball teams.
Tuned for tournament prediction with margin-of-victory adjustment.
"""

import numpy as np
import pandas as pd
from configs.config import ELO_CONFIG


class EloRatingSystem:
    """Custom Elo rating system for NCAA basketball."""

    def __init__(self, config=None):
        cfg = config or ELO_CONFIG
        self.initial_rating = cfg["initial_rating"]
        self.k_factor = cfg["k_factor"]
        self.k_factor_tournament = cfg["k_factor_tournament"]
        self.home_advantage = cfg["home_advantage"]
        self.margin_multiplier = cfg["margin_multiplier"]
        self.season_reversion = cfg["season_reversion"]
        self.ratings = {}

    def get_rating(self, team_id):
        """Get current Elo rating for a team."""
        return self.ratings.get(team_id, self.initial_rating)

    def expected_score(self, rating_a, rating_b):
        """Calculate expected win probability for team A."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def margin_of_victory_multiplier(self, margin, elo_diff):
        """Adjust K-factor based on margin of victory."""
        return np.log(abs(margin) + 1) * (2.2 / ((elo_diff * self.margin_multiplier) + 2.2))

    def update(self, winner_id, loser_id, margin, is_home_winner=None, is_tournament=False):
        """Update ratings after a game."""
        winner_rating = self.get_rating(winner_id)
        loser_rating = self.get_rating(loser_id)

        # Home court adjustment
        if is_home_winner is True:
            winner_adj = winner_rating + self.home_advantage
            loser_adj = loser_rating
        elif is_home_winner is False:
            winner_adj = winner_rating
            loser_adj = loser_rating + self.home_advantage
        else:  # Neutral
            winner_adj = winner_rating
            loser_adj = loser_rating

        expected_w = self.expected_score(winner_adj, loser_adj)

        k = self.k_factor_tournament if is_tournament else self.k_factor
        mov_mult = self.margin_of_victory_multiplier(margin, abs(winner_rating - loser_rating))
        adjustment = k * mov_mult * (1 - expected_w)

        self.ratings[winner_id] = winner_rating + adjustment
        self.ratings[loser_id] = loser_rating - adjustment

    def new_season(self):
        """Revert ratings toward mean between seasons."""
        for team_id in self.ratings:
            self.ratings[team_id] = (
                self.ratings[team_id] * (1 - self.season_reversion)
                + self.initial_rating * self.season_reversion
            )

    def predict(self, team_a_id, team_b_id):
        """Predict P(team_a beats team_b) on neutral court."""
        return self.expected_score(
            self.get_rating(team_a_id),
            self.get_rating(team_b_id)
        )

    def build_ratings(self, regular_season_df, tourney_df=None, start_season=None):
        """
        Build Elo ratings from historical game results.

        Parameters
        ----------
        regular_season_df : pd.DataFrame
            Must have: Season, WTeamID, LTeamID, WScore, LScore, WLoc
        tourney_df : pd.DataFrame, optional
            Tournament results (same columns). Only used for updating Elo,
            NOT for building features (to avoid data leakage).
        start_season : int, optional
            First season to process.
        """
        start = start_season or ELO_CONFIG["start_season"]
        self.ratings = {}

        all_games = regular_season_df.copy()
        if tourney_df is not None:
            all_games = pd.concat([all_games, tourney_df], ignore_index=True)
        all_games = all_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

        seasons = sorted(all_games["Season"].unique())
        season_ratings = {}

        for season in seasons:
            if season < start:
                continue

            if season > start:
                self.new_season()

            season_games = all_games[all_games["Season"] == season]
            is_tourney = tourney_df is not None

            for _, game in season_games.iterrows():
                margin = game["WScore"] - game["LScore"]
                is_home = game.get("WLoc", "N")
                home_winner = True if is_home == "H" else (False if is_home == "A" else None)

                # Check if this is a tournament game
                tournament_game = False
                if tourney_df is not None:
                    tournament_game = (
                        (game["Season"] == season)
                        and game.get("DayNum", 0) > 132  # Tournament starts ~day 134
                    )

                self.update(
                    game["WTeamID"],
                    game["LTeamID"],
                    margin,
                    is_home_winner=home_winner,
                    is_tournament=tournament_game,
                )

            # Snapshot end-of-regular-season ratings
            season_ratings[season] = dict(self.ratings)

        return season_ratings

    def get_season_ratings_df(self, season_ratings):
        """Convert season ratings dict to DataFrame."""
        rows = []
        for season, ratings in season_ratings.items():
            for team_id, rating in ratings.items():
                rows.append({
                    "Season": season,
                    "TeamID": team_id,
                    "EloRating": rating,
                })
        return pd.DataFrame(rows)
