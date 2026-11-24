import os
from pathlib import Path

import requests

BASE_URL = "https://api.football-data.org/v4"
DEFAULT_ENDPOINT = "/competitions/PD/matches"


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class FootballAPIClient:

    def __init__(self, base_url: str = BASE_URL, api_key: str | None = None, timeout: int = 15) -> None:
        project_root = Path(__file__).resolve().parent.parent
        load_env_file(project_root / ".env")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("FOOTBALL_API_KEY", "")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["X-Auth-Token"] = self.api_key
        return headers

    def fetch_matches(self, endpoint: str = DEFAULT_ENDPOINT, params: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self._headers(), params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    client = FootballAPIClient()
    payload = client.fetch_matches()
    matches = payload.get("matches", [])
    print(f"Total meciuri primite: {len(matches)}")
    for match in matches[:5]:
        home_team = match.get("homeTeam", {}).get("name", "N/A")
        away_team = match.get("awayTeam", {}).get("name", "N/A")
        utc_date = match.get("utcDate", "N/A")
        status = match.get("status", "N/A")
        print(f"- [{status}] {utc_date} | {home_team} vs {away_team}")