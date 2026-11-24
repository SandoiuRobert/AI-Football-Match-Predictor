import sqlite3
from pathlib import Path

import streamlit as st

from model import DEFAULT_DB_PATH, MatchOutcomePredictor

st.set_page_config(
    page_title="AI Football Match Predictor",
    page_icon="⚽",
    layout="wide",
)


@st.cache_resource
def load_predictor(db_path: Path) -> MatchOutcomePredictor:
    predictor = MatchOutcomePredictor(db_path=db_path)
    predictor.fit()
    return predictor


def get_available_teams(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    query = """
            SELECT DISTINCT team_name FROM (
                                               SELECT home_team AS team_name FROM matches WHERE home_team IS NOT NULL
                                               UNION
                                               SELECT away_team AS team_name FROM matches WHERE away_team IS NOT NULL
                                           )
            ORDER BY team_name ASC \
            """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def main() -> None:
    st.title("⚽ AI Football Match Predictor")
    st.caption("Predicții 1X2 bazate pe istoricul meciurilor din LaLiga 2026-2027")

    db_path = DEFAULT_DB_PATH
    if not db_path.exists():
        st.error(f"Nu am gasit baza de date la `{db_path}`. Rulează mai intai `src/database.py`.")
        return

    teams = get_available_teams(db_path)
    if len(teams) < 2:
        st.warning("Nu exista suficiente echipe in baza de date pentru a selecta un meci.")
        return

    try:
        predictor = load_predictor(db_path)
    except (ValueError, sqlite3.Error) as exc:
        st.error(f"Eroare la antrenarea modelului: {exc}")
        return

    trained_matches = len(predictor.history_df)
    st.info(f"Model antrenat pe **{trained_matches}** meciuri finalizate din La Liga.")

    col_home, col_away = st.columns(2)
    default_home_idx = teams.index("FC Barcelona") if "FC Barcelona" in teams else 0

    with col_home:
        home_team = st.selectbox("Echipa gazda (1)", options=teams, index=default_home_idx)

    with col_away:
        away_options = [t for t in teams if t != home_team]
        away_default = away_options.index("Real Madrid CF") if "Real Madrid CF" in away_options else 0
        away_team = st.selectbox("Echipa oaspete (2)", options=away_options, index=away_default)

    prediction = predictor.predict_match_1x2(home_team, away_team)

    st.divider()
    st.subheader(f"{home_team} vs {away_team}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Victorie Gazda (1 / H)", f"{prediction['prob_home_1']}%")
    m2.metric("Egal (X / D)", f"{prediction['prob_draw_X']}%")
    m3.metric("Victorie Oaspete (2 / A)", f"{prediction['prob_away_2']}%")

    outcome_labels = {
        "H": f"Victorie {home_team} (1)",
        "D": "Egal (X)",
        "A": f"Victorie {away_team} (2)",
    }
    recommended = outcome_labels.get(prediction["predicted_outcome"], prediction["predicted_outcome"])
    st.success(f"**Rezultat probabil indicat de model:** {recommended}")

    chart_data = {
        "Rezultat": ["1 (Gazda)", "X (Egal)", "2 (Oaspete)"],
        "Probabilitate (%)": [
            prediction["prob_home_1"],
            prediction["prob_draw_X"],
            prediction["prob_away_2"],
        ],
    }
    st.bar_chart(chart_data, x="Rezultat", y="Probabilitate (%)", height=280)


if __name__ == "__main__":
    main()