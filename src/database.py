import sqlite3
from pathlib import Path
from typing import Any

from fetcher import FootballAPIClient

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football.db"


class MatchDatabase:

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def initialize_schema(self) -> None:
        query = """
                CREATE TABLE IF NOT EXISTS matches (
                                                       match_id INTEGER PRIMARY KEY,
                                                       competition_code TEXT,
                                                       season TEXT,
                                                       utc_date TEXT,
                                                       status TEXT,
                                                       matchday INTEGER,
                                                       home_team TEXT NOT NULL,
                                                       away_team TEXT NOT NULL,
                                                       home_goals INTEGER,
                                                       away_goals INTEGER,
                                                       full_time_result TEXT
                ) \
                """
        with self._connect() as conn:
            conn.execute(query)
            conn.commit()

    @staticmethod
    def _derive_result(home_goals: int | None, away_goals: int | None) -> str | None:
        if home_goals is None or away_goals is None:
            return None
        if home_goals > away_goals:
            return "H"
        if away_goals > home_goals:
            return "A"
        return "D"

    def upsert_matches(self, matches_payload: list[dict[str, Any]], competition_code: str = "PD") -> int:
        rows: list[tuple[Any, ...]] = []
        for match in matches_payload:
            match_id = match.get("id")
            if match_id is None:
                continue

            season_obj = match.get("season") or {}
            start_date = season_obj.get("startDate", "")
            end_date = season_obj.get("endDate", "")
            season_label = f"{start_date[:4]}/{end_date[:4]}" if start_date and end_date else None

            home_team = (match.get("homeTeam") or {}).get("name") or "Unknown Home"
            away_team = (match.get("awayTeam") or {}).get("name") or "Unknown Away"

            score_obj = match.get("score") or {}
            full_time = score_obj.get("fullTime") or {}
            home_goals = full_time.get("home")
            away_goals = full_time.get("away")
            result = self._derive_result(home_goals, away_goals)

            rows.append(
                (
                    match_id,
                    competition_code,
                    season_label,
                    match.get("utcDate"),
                    match.get("status"),
                    match.get("matchday"),
                    home_team,
                    away_team,
                    home_goals,
                    away_goals,
                    result,
                )
            )

        insert_query = """
                       INSERT INTO matches (
                           match_id, competition_code, season, utc_date, status,
                           matchday, home_team, away_team, home_goals, away_goals, full_time_result
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(match_id) DO UPDATE SET
                           competition_code = excluded.competition_code,
                                                        season = excluded.season,
                                                        utc_date = excluded.utc_date,
                                                        status = excluded.status,
                                                        matchday = excluded.matchday,
                                                        home_team = excluded.home_team,
                                                        away_team = excluded.away_team,
                                                        home_goals = excluded.home_goals,
                                                        away_goals = excluded.away_goals,
                                                        full_time_result = excluded.full_time_result \
                       """
        with self._connect() as conn:
            conn.executemany(insert_query, rows)
            conn.commit()
        return len(rows)

    def count_matches(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM matches")
            row = cursor.fetchone()
            return int(row[0]) if row else 0


if __name__ == "__main__":
    client = FootballAPIClient()
    payload = client.fetch_matches()
    db = MatchDatabase()
    saved = db.upsert_matches(payload.get("matches", []), competition_code="PD")
    print(f"Au fost procesate {saved} meciuri. Total în SQLite ({db.db_path}): {db.count_matches()}")