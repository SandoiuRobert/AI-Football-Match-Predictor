import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football.db"
CLASSES = ["A", "D", "H"]


class MatchOutcomePredictor:

    def __init__(self, db_path: Path = DEFAULT_DB_PATH, form_window: int = 5) -> None:
        self.db_path = db_path
        self.form_window = form_window
        self.pipeline: Pipeline | None = None
        self.history_df: pd.DataFrame = pd.DataFrame()

    def load_finished_matches(self) -> pd.DataFrame:
        query = """
                SELECT
                    match_id,
                    utc_date,
                    home_team,
                    away_team,
                    home_goals,
                    away_goals,
                    full_time_result
                FROM matches
                WHERE status = 'FINISHED'
                  AND full_time_result IN ('H', 'D', 'A')
                ORDER BY utc_date ASC, match_id ASC \
                """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        return df

    def _build_features_table(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        records: list[dict[str, float]] = []
        targets: list[str] = []

        team_points: dict[str, list[float]] = {}
        team_gf: dict[str, list[float]] = {}
        team_ga: dict[str, list[float]] = {}
        h2h_home_points: dict[tuple[str, str], list[float]] = {}

        for row in df.itertuples(index=False):
            home = str(row.home_team)
            away = str(row.away_team)
            result = str(row.full_time_result)

            h_pts_hist = team_points.get(home, [])
            a_pts_hist = team_points.get(away, [])
            h_gf_hist = team_gf.get(home, [])
            a_gf_hist = team_gf.get(away, [])
            h_ga_hist = team_ga.get(home, [])
            a_ga_hist = team_ga.get(away, [])

            h_form = float(np.mean(h_pts_hist[-self.form_window :])) if h_pts_hist else 1.0
            a_form = float(np.mean(a_pts_hist[-self.form_window :])) if a_pts_hist else 1.0

            h_avg_gf = float(np.mean(h_gf_hist[-self.form_window :])) if h_gf_hist else 1.0
            h_avg_ga = float(np.mean(h_ga_hist[-self.form_window :])) if h_ga_hist else 1.0
            a_avg_gf = float(np.mean(a_gf_hist[-self.form_window :])) if a_gf_hist else 1.0
            a_avg_ga = float(np.mean(a_ga_hist[-self.form_window :])) if a_ga_hist else 1.0

            pair_key = (home, away)
            h2h_hist = h2h_home_points.get(pair_key, [])
            h2h_avg = float(np.mean(h2h_hist)) if h2h_hist else 1.0

            records.append(
                {
                    "home_form": h_form,
                    "away_form": a_form,
                    "form_diff": h_form - a_form,
                    "home_avg_gf": h_avg_gf,
                    "home_avg_ga": h_avg_ga,
                    "away_avg_gf": a_avg_gf,
                    "away_avg_ga": a_avg_ga,
                    "h2h_home_pts_avg": h2h_avg,
                }
            )
            targets.append(result)

            if result == "H":
                h_pts, a_pts = 3.0, 0.0
            elif result == "A":
                h_pts, a_pts = 0.0, 3.0
            else:
                h_pts, a_pts = 1.0, 1.0

            hg = float(row.home_goals) if row.home_goals is not None else 0.0
            ag = float(row.away_goals) if row.away_goals is not None else 0.0

            team_points.setdefault(home, []).append(h_pts)
            team_points.setdefault(away, []).append(a_pts)
            team_gf.setdefault(home, []).append(hg)
            team_ga.setdefault(home, []).append(ag)
            team_gf.setdefault(away, []).append(ag)
            team_ga.setdefault(away, []).append(hg)
            h2h_home_points.setdefault(pair_key, []).append(h_pts)

        return pd.DataFrame(records), pd.Series(targets)

    def fit(self) -> int: #antreneaza modelul cu meciuri finalizate
        self.history_df = self.load_finished_matches()
        if len(self.history_df) < 10:
            raise ValueError(
                f"Prea putine meciuri finalizate in baza de date ({len(self.history_df)}) pentru antrenare."
            )

        X, y = self._build_features_table(self.history_df)
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        self.pipeline.fit(X, y)
        return len(self.history_df)

    def _team_snapshot(self, team_name: str) -> dict[str, float]:
        if self.history_df.empty:
            return {"form": 1.0, "gf": 1.0, "ga": 1.0}

        pts_list: list[float] = []
        gf_list: list[float] = []
        ga_list: list[float] = []

        for row in self.history_df.itertuples(index=False):
            if row.home_team == team_name:
                gf_list.append(float(row.home_goals or 0))
                ga_list.append(float(row.away_goals or 0))
                if row.full_time_result == "H":
                    pts_list.append(3.0)
                elif row.full_time_result == "D":
                    pts_list.append(1.0)
                else:
                    pts_list.append(0.0)
            elif row.away_team == team_name:
                gf_list.append(float(row.away_goals or 0))
                ga_list.append(float(row.home_goals or 0))
                if row.full_time_result == "A":
                    pts_list.append(3.0)
                elif row.full_time_result == "D":
                    pts_list.append(1.0)
                else:
                    pts_list.append(0.0)

        if not pts_list:
            return {"form": 1.0, "gf": 1.0, "ga": 1.0}

        return {
            "form": float(np.mean(pts_list[-self.form_window :])),
            "gf": float(np.mean(gf_list[-self.form_window :])),
            "ga": float(np.mean(ga_list[-self.form_window :])),
        }

    def _h2h_snapshot(self, home_team: str, away_team: str) -> float:
        if self.history_df.empty:
            return 1.0

        pts: list[float] = []
        for row in self.history_df.itertuples(index=False):
            if row.home_team == home_team and row.away_team == away_team:
                if row.full_time_result == "H":
                    pts.append(3.0)
                elif row.full_time_result == "D":
                    pts.append(1.0)
                else:
                    pts.append(0.0)
            elif row.home_team == away_team and row.away_team == home_team:
                if row.full_time_result == "A":
                    pts.append(3.0)
                elif row.full_time_result == "D":
                    pts.append(1.0)
                else:
                    pts.append(0.0)

        return float(np.mean(pts)) if pts else 1.0

    def predict_match_1x2(self, home_team: str, away_team: str) -> dict[str, Any]:
        if self.pipeline is None:
            self.fit()

        h_snap = self._team_snapshot(home_team)
        a_snap = self._team_snapshot(away_team)
        h2h_avg = self._h2h_snapshot(home_team, away_team)

        features = pd.DataFrame(
            [
                {
                    "home_form": h_snap["form"],
                    "away_form": a_snap["form"],
                    "form_diff": h_snap["form"] - a_snap["form"],
                    "home_avg_gf": h_snap["gf"],
                    "home_avg_ga": h_snap["ga"],
                    "away_avg_gf": a_snap["gf"],
                    "away_avg_ga": a_snap["ga"],
                    "h2h_home_pts_avg": h2h_avg,
                }
            ]
        )

        assert self.pipeline is not None
        probs = self.pipeline.predict_proba(features)[0]
        class_labels = list(self.pipeline.classes_)
        prob_map = {label: float(p) for label, p in zip(class_labels, probs, strict=False)}

        best_label = max(prob_map, key=lambda k: prob_map[k])
        return {
            "home_team": home_team,
            "away_team": away_team,
            "prob_home_1": round(prob_map.get("H", 0.0) * 100, 1),
            "prob_draw_X": round(prob_map.get("D", 0.0) * 100, 1),
            "prob_away_2": round(prob_map.get("A", 0.0) * 100, 1),
            "predicted_outcome": best_label,
        }


if __name__ == "__main__":
    predictor = MatchOutcomePredictor()
    trained_count = predictor.fit()
    print(f"Model antrenat pe {trained_count} meciuri finalizate din SQLite.")

    # Exemplu de predicție pe două echipe din baza de date
    sample = predictor.predict_match_1x2("FC Barcelona", "Real Madrid CF")
    print(sample)