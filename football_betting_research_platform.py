"""Football Betting Research Platform (MVP)

Run locally:
  pip install streamlit pandas numpy scipy requests
  streamlit run football_betting_research_platform.py

This educational research tool does not place bets or automate bookmaker actions.
Can load data from: CSV upload, live APIs (football-data.org, the-odds-api.com), or demo data.
Required columns (case-insensitive aliases accepted): Date, HomeTeam, AwayTeam, FTHG, FTAG.
Optional odds columns: B365H, B365D, B365A (or OddsHome, OddsDraw, OddsAway).
"""

from __future__ import annotations

import io
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

try:
    from data_sources import FootballDataConnector, OddsConnector
except ImportError:
    st.error("Missing data_sources.py module. Please create it with the provided code.")
    st.stop()

st.set_page_config(page_title="Football Betting Research Lab", page_icon="⚽", layout="wide")

REQUIRED_ALIASES = {
    "date": ["date", "matchdate", "match_date"],
    "home": ["hometeam", "home_team", "home", "team_home"],
    "away": ["awayteam", "away_team", "away", "team_away"],
    "home_goals": ["fthg", "homegoals", "home_goals", "hg", "score_home"],
    "away_goals": ["ftag", "awaygoals", "away_goals", "ag", "score_away"],
}
ODDS_ALIASES = {
    "home_odds": ["b365h", "oddshome", "homeodds", "odds_home", "1"],
    "draw_odds": ["b365d", "oddsdraw", "drawodds", "odds_draw", "x"],
    "away_odds": ["b365a", "oddsaway", "awayodds", "odds_away", "2"],
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to canonical names using aliases"""
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    mapping = {}
    for canonical, aliases in {**REQUIRED_ALIASES, **ODDS_ALIASES}.items():
        for alias in aliases:
            if alias in lookup:
                mapping[lookup[alias]] = canonical
                break
    out = df.rename(columns=mapping).copy()
    return out


def prepare_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Prepare and validate matches data"""
    df = normalise_columns(raw)
    missing = [c for c in REQUIRED_ALIASES if c not in df.columns]
    if missing:
        raise ValueError("Chybí povinné sloupce: " + ", ".join(missing))
    keep = list(REQUIRED_ALIASES) + [c for c in ODDS_ALIASES if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    for col in ["home_goals", "away_goals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["home_odds", "draw_odds", "away_odds"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["home"] = df["home"].astype(str).str.strip()
    df["away"] = df["away"].astype(str).str.strip()
    df = df.dropna(subset=["home", "away", "home_goals", "away_goals"])
    df = df[(df.home_goals >= 0) & (df.away_goals >= 0) & (df.home != df.away)]
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return df.sort_values("date", na_position="last").reset_index(drop=True)


def league_summary(df: pd.DataFrame) -> dict:
    """Calculate league-level statistics"""
    return {
        "matches": len(df),
        "teams": pd.unique(pd.concat([df.home, df.away])).size,
        "home_avg": df.home_goals.mean(),
        "away_avg": df.away_goals.mean(),
    }


def team_ratings(df: pd.DataFrame, shrinkage: float = 5.0) -> pd.DataFrame:
    """Calculate team attack/defence ratings with shrinkage"""
    s = league_summary(df)
    home_avg, away_avg = s["home_avg"], s["away_avg"]
    teams = sorted(set(df.home) | set(df.away))
    rows = []
    for team in teams:
        h = df[df.home == team]
        a = df[df.away == team]
        def smooth(total: float, n: int, baseline: float) -> float:
            return (total + shrinkage * baseline) / (n + shrinkage)
        home_for = smooth(h.home_goals.sum(), len(h), home_avg)
        home_against = smooth(h.away_goals.sum(), len(h), away_avg)
        away_for = smooth(a.away_goals.sum(), len(a), away_avg)
        away_against = smooth(a.home_goals.sum(), len(a), home_avg)
        rows.append({
            "team": team,
            "home_matches": len(h),
            "away_matches": len(a),
            "home_attack": home_for / home_avg if home_avg else 1,
            "home_defence": home_against / away_avg if away_avg else 1,
            "away_attack": away_for / away_avg if away_avg else 1,
            "away_defence": away_against / home_avg if home_avg else 1,
        })
    return pd.DataFrame(rows).set_index("team")


def expected_goals(home: str, away: str, ratings: pd.DataFrame, summary: dict) -> tuple[float, float]:
    """Calculate expected goals using Poisson parameters"""
    h, a = ratings.loc[home], ratings.loc[away]
    lam_home = summary["home_avg"] * h.home_attack * a.away_defence
    lam_away = summary["away_avg"] * a.away_attack * h.home_defence
    return float(lam_home), float(lam_away)


def probability_matrix(lam_home: float, lam_away: float, max_goals: int = 8) -> pd.DataFrame:
    """Generate Poisson probability matrix for outcomes"""
    goals = np.arange(max_goals + 1)
    values = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
    return pd.DataFrame(values, index=goals, columns=goals)


def market_probabilities(matrix: pd.DataFrame) -> dict:
    """Calculate market outcome probabilities from matrix"""
    values = matrix.to_numpy()
    return {
        "1": float(np.tril(values, -1).sum()),
        "X": float(np.trace(values)),
        "2": float(np.triu(values, 1).sum()),
        "Over 2.5": float(sum(values[i, j] for i in range(values.shape[0]) for j in range(values.shape[1]) if i + j >= 3)),
        "BTTS Ano": float(sum(values[i, j] for i in range(1, values.shape[0]) for j in range(1, values.shape[1]))),
        "Pokrytá hmata": float(values.sum()),
    }


def no_vig_probs(odds: list[float]) -> list[float]:
    """Convert bookmaker odds to no-vig probabilities"""
    implied = np.array([1 / x for x in odds], dtype=float)
    return list(implied / implied.sum())


def ev_pct(probability: float, odds: float) -> float:
    """Calculate expected value percentage"""
    return probability * odds - 1


def kelly_fraction(probability: float, odds: float, fraction: float) -> float:
    """Calculate Kelly criterion stake fraction"""
    b = odds - 1
    full = max(0.0, (b * probability - (1 - probability)) / b)
    return full * fraction


def fmt_pct(x: float) -> str:
    """Format float as percentage"""
    return f"{x * 100:.2f}%"


def init_state() -> None:
    """Initialize session state for paper-bet ledger"""
    if "ledger" not in st.session_state:
        st.session_state.ledger = pd.DataFrame(columns=[
            "created_at", "home", "away", "market", "odds_taken", "model_probability",
            "no_vig_probability", "edge", "ev_pct", "stake", "status", "result", "profit_loss"
        ])


init_state()
st.title("⚽ Football Betting Research Lab")
st.caption("Výzkumný MVP pro modelování pravděpodobností a paper betting. Bez napojení na sázkové kanceláře a bez automatického sázení.")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    st.header("📡 Data Source")
    data_source = st.radio(
        "Select data input method",
        ["Upload CSV", "Live API", "Demo Data"],
        help="Choose where to load match data from"
    )
    
    st.divider()
    st.header("⚙️ Model Settings")
    shrinkage = st.slider(
        "Shrinkage (sample stabilization)",
        0.0, 20.0, 5.0, 0.5,
        help="Higher values pull team stats toward league average"
    )
    max_goals = st.slider("Max goals in matrix", 5, 12, 8)
    
    st.divider()
    st.header("💰 Risk Framework")
    bankroll = st.number_input("Paper bankroll", min_value=100.0, value=10000.0, step=500.0)
    kelly_part = st.selectbox(
        "Kelly fraction",
        [0.0, 0.25, 0.5],
        index=1,
        format_func=lambda x: "Flat stake" if x == 0 else f"{x:.0%} Kelly"
    )
    flat_pct = st.slider("Flat stake (% of bankroll)", 0.25, 3.0, 1.0, 0.25) / 100
    stake_cap_pct = st.slider("Max stake (% of bankroll)", 0.5, 5.0, 2.0, 0.5) / 100
    min_edge = st.slider("Minimum model edge", 0.00, 0.15, 0.03, 0.005)
    
    if data_source == "Live API":
        st.divider()
        st.header("🌍 League Selection")
        league_name = st.selectbox(
            "Select league",
            ["Premier League (PL)", "La Liga (SA)", "Bundesliga (BL1)", "Serie A (SA)", "Ligue 1 (FL1)"],
            help="Select which league to fetch data from"
        )
        league_code = league_name.split("(")[1].rstrip(")")
        days_back = st.slider("Historical data (days back)", 7, 90, 30)

# ============================================================================
# DATA LOADING
# ============================================================================
st.header("1. Historical Data")

matches = None

if data_source == "Upload CSV":
    upload = st.file_uploader("Upload CSV with finished matches", type="csv")
    if upload is None:
        st.info("Upload a CSV file with required columns: Date, HomeTeam, AwayTeam, FTHG, FTAG")
        st.code(
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
            "2025-08-10,Team A,Team B,2,1,1.95,3.60,4.10",
            language="text"
        )
        st.stop()
    
    try:
        raw = pd.read_csv(upload)
        matches = prepare_matches(raw)
        st.success(f"✓ Loaded {len(matches)} valid matches from CSV")
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

elif data_source == "Live API":
    st.subheader("📡 Live Data & Odds")
    
    with st.spinner("Fetching data from APIs..."):
        connector = FootballDataConnector()
        odds_connector = OddsConnector()
        
        col_live1, col_live2 = st.columns(2)
        
        with col_live1:
            st.write("**Recently Completed Matches**")
            completed = connector.fetch_completed_matches(league=league_code, days_back=days_back)
            if not completed.empty:
                try:
                    matches = prepare_matches(completed)
                    st.success(f"✓ Loaded {len(matches)} matches")
                except Exception as e:
                    st.warning(f"Could not prepare matches: {e}")
            else:
                st.warning("No recent completed matches found")
        
        with col_live2:
            st.write("**Upcoming Matches**")
            upcoming = connector.fetch_upcoming_matches(league=league_code)
            if not upcoming.empty:
                st.dataframe(upcoming[["home", "away", "date"]], use_container_width=True, hide_index=True)
            else:
                st.info("No upcoming matches found")
        
        st.divider()
        st.write("**Live Odds (Multiple Bookmakers)**")
        live_odds = odds_connector.fetch_live_odds()
        if not live_odds.empty:
            st.dataframe(live_odds, use_container_width=True, hide_index=True)
        else:
            st.info("No live odds available")

elif data_source == "Demo Data":
    st.info("Using demo data for testing. Data refreshes on each run.")
    base_date = datetime.utcnow()
    dates = [base_date - timedelta(days=i) for i in range(1, 31)]
    teams_pool = [
        "Manchester United", "Liverpool", "Arsenal", "Chelsea",
        "Manchester City", "Tottenham", "Brighton", "Newcastle",
        "Fulham", "Brentford", "Aston Villa", "Wolverhampton"
    ]
    
    records = []
    for i, date in enumerate(dates):
        home_idx = (i * 2) % len(teams_pool)
        away_idx = (i * 2 + 1) % len(teams_pool)
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "home": teams_pool[home_idx],
            "away": teams_pool[away_idx],
            "home_goals": i % 4,
            "away_goals": (i + 1) % 4,
        })
    
    demo_df = pd.DataFrame(records)
    try:
        matches = prepare_matches(demo_df)
        st.success(f"✓ Loaded {len(matches)} demo matches")
    except Exception as e:
        st.error(f"Failed to prepare demo data: {e}")
        st.stop()

# Check if we have data
if matches is None or matches.empty:
    st.error("No valid match data available. Please load data using one of the methods above.")
    st.stop()

if len(matches) < 50:
    st.warning("Dataset contains fewer than 50 matches. Results will be very unstable; use more seasons for better accuracy.")

# ============================================================================
# LEAGUE SUMMARY & TEAM RATINGS
# ============================================================================
summary = league_summary(matches)
ratings = team_ratings(matches, shrinkage)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Valid Matches", summary["matches"])
c2.metric("Teams", summary["teams"])
c3.metric("Avg Home Goals", f"{summary['home_avg']:.2f}")
c4.metric("Avg Away Goals", f"{summary['away_avg']:.2f}")

with st.expander("Team Ratings & Data Validation"):
    st.write("**Team Attack/Defence Ratings**")
    st.dataframe(
        ratings.sort_values("home_attack", ascending=False),
        use_container_width=True
    )
    st.write("**Latest 20 Matches**")
    st.dataframe(matches.tail(20), use_container_width=True)

# ============================================================================
# MATCH PREDICTION
# ============================================================================
st.header("2. Match Prediction")
teams = sorted(ratings.index.tolist())

col_a, col_b = st.columns(2)
home = col_a.selectbox("Home Team", teams, index=0)
away_candidates = [t for t in teams if t != home]
away = col_b.selectbox("Away Team", away_candidates, index=0)

lam_h, lam_a = expected_goals(home, away, ratings, summary)
matrix = probability_matrix(lam_h, lam_a, max_goals)
probs = market_probabilities(matrix)

p1, px, p2 = probs["1"], probs["X"], probs["2"]
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("λ Home", f"{lam_h:.2f}")
mc2.metric("λ Away", f"{lam_a:.2f}")
mc3.metric("Most Likely 1X2", max({"1": p1, "X": px, "2": p2}, key={"1": p1, "X": px, "2": p2}.get))
mc4.metric("Matrix Coverage", fmt_pct(probs["Pokrytá hmata"]))

outcome_table = pd.DataFrame({
    "Outcome": ["1", "X", "2", "Over 2.5", "BTTS Yes"],
    "Model Probability": [p1, px, p2, probs["Over 2.5"], probs["BTTS Ano"]],
})
outcome_table["Model Probability"] = outcome_table["Model Probability"].map(fmt_pct)
outcome_table["Fair Odds"] = [f"{1 / p:.3f}" if p > 0 else "—" for p in [p1, px, p2, probs["Over 2.5"], probs["BTTS Ano"]]]
st.dataframe(outcome_table, use_container_width=True, hide_index=True)

with st.expander("Scoreline Matrix (rows=home goals, cols=away goals)"):
    st.dataframe(matrix.style.format("{:.2%}"), use_container_width=True)

# ============================================================================
# MARKET COMPARISON & PAPER BET
# ============================================================================
st.header("3. Market Comparison & Paper Bet")
st.caption("Enter the odds you actually observed. The system records only research data—it places no actual bets.")

has_historical_odds = all(c in matches.columns for c in ["home_odds", "draw_odds", "away_odds"])
defaults = [2.00, 3.50, 3.75]
if has_historical_odds:
    latest = matches.dropna(subset=["home_odds", "draw_odds", "away_odds"]).tail(1)
    if not latest.empty:
        defaults = [float(latest.iloc[0][c]) for c in ["home_odds", "draw_odds", "away_odds"]]

oc1, oc2, oc3 = st.columns(3)
oh = oc1.number_input("Odds 1 (Home)", min_value=1.01, value=defaults[0], step=0.01)
od = oc2.number_input("Odds X (Draw)", min_value=1.01, value=defaults[1], step=0.01)
oa = oc3.number_input("Odds 2 (Away)", min_value=1.01, value=defaults[2], step=0.01)

market_odds = [oh, od, oa]
market_no_vig = no_vig_probs(market_odds)

market_df = pd.DataFrame({
    "Outcome": ["1", "X", "2"],
    "Market Odds": market_odds,
    "Model p": [p1, px, p2],
    "No-vig Market p": market_no_vig,
})
market_df["Edge vs No-vig"] = market_df["Model p"] - market_df["No-vig Market p"]
market_df["EV %"] = [ev_pct(p, o) for p, o in zip([p1, px, p2], market_odds)]
market_df["Model Fair Odds"] = [1 / p for p in [p1, px, p2]]

for c in ["Model p", "No-vig Market p", "Edge vs No-vig", "EV %"]:
    market_df[c] = market_df[c].map(fmt_pct)

st.dataframe(market_df, use_container_width=True, hide_index=True)

market_label = st.selectbox("Select outcome to paper bet", ["1", "X", "2"])
index = {"1": 0, "X": 1, "2": 2}[market_label]
selected_p = [p1, px, p2][index]
selected_odds = market_odds[index]
selected_no_vig = market_no_vig[index]
edge = selected_p - selected_no_vig
ev = ev_pct(selected_p, selected_odds)

if kelly_part:
    calculated_stake = bankroll * kelly_fraction(selected_p, selected_odds, kelly_part)
else:
    calculated_stake = bankroll * flat_pct

stake = min(calculated_stake, bankroll * stake_cap_pct)

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Model p", fmt_pct(selected_p))
sc2.metric("Edge vs No-vig", fmt_pct(edge))
sc3.metric("EV", fmt_pct(ev))
sc4.metric("Suggested Paper Stake", f"{stake:,.0f}")

qualifies = edge >= min_edge and ev > 0
if qualifies:
    st.success("✓ Candidate meets your threshold. This is research—not a guaranteed profit signal.")
else:
    st.warning("✗ Candidate does not meet your threshold or has negative EV. Skip this bet in research mode.")

if st.button("Add to Paper-Bet Ledger", type="primary"):
    entry = pd.DataFrame([{
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "home": home,
        "away": away,
        "market": market_label,
        "odds_taken": selected_odds,
        "model_probability": selected_p,
        "no_vig_probability": selected_no_vig,
        "edge": edge,
        "ev_pct": ev,
        "stake": stake,
        "status": "open",
        "result": "",
        "profit_loss": np.nan,
    }])
    st.session_state.ledger = pd.concat([st.session_state.ledger, entry], ignore_index=True)
    st.success("✓ Record added to paper-bet ledger (session only)")

# ============================================================================
# PAPER-BET LEDGER
# ============================================================================
st.header("4. Paper-Bet Ledger")
ledger = st.session_state.ledger.copy()

if ledger.empty:
    st.info("No records yet. Add a paper bet above to start tracking.")
else:
    display = ledger.copy()
    for c in ["model_probability", "no_vig_probability", "edge", "ev_pct"]:
        display[c] = display[c].astype(float).map(fmt_pct)
    st.dataframe(display, use_container_width=True, hide_index=True)
    
    st.download_button(
        "Download Ledger as CSV",
        data=ledger.to_csv(index=False).encode("utf-8"),
        file_name="paper_bet_ledger.csv",
        mime="text/csv",
    )

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption(
    "**Important:** This platform is an educational analytical tool. It does not circumvent bookmaker "
    "systems, does not place automatic bets, and does not replace legal or financial advice. "
    "All paper bets are for research purposes only."
)
