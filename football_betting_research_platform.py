"""Football Betting Research Platform (MVP)

Run locally:
  pip install streamlit pandas numpy scipy
  streamlit run football_betting_research_platform.py

This educational research tool does not place bets or automate bookmaker actions.
Upload a CSV containing historical finished matches. Required columns (case-insensitive
aliases are accepted): Date, HomeTeam, AwayTeam, FTHG, FTAG.
Optional odds columns: B365H, B365D, B365A (or OddsHome, OddsDraw, OddsAway).
"""

from __future__ import annotations

import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

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
    return {
        "matches": len(df),
        "teams": pd.unique(pd.concat([df.home, df.away])).size,
        "home_avg": df.home_goals.mean(),
        "away_avg": df.away_goals.mean(),
    }


def team_ratings(df: pd.DataFrame, shrinkage: float = 5.0) -> pd.DataFrame:
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
    h, a = ratings.loc[home], ratings.loc[away]
    lam_home = summary["home_avg"] * h.home_attack * a.away_defence
    lam_away = summary["away_avg"] * a.away_attack * h.home_defence
    return float(lam_home), float(lam_away)


def probability_matrix(lam_home: float, lam_away: float, max_goals: int = 8) -> pd.DataFrame:
    goals = np.arange(max_goals + 1)
    values = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
    return pd.DataFrame(values, index=goals, columns=goals)


def market_probabilities(matrix: pd.DataFrame) -> dict:
    values = matrix.to_numpy()
    return {
        "1": float(np.tril(values, -1).sum()),
        "X": float(np.trace(values)),
        "2": float(np.triu(values, 1).sum()),
        "Over 2.5": float(sum(values[i, j] for i in range(values.shape[0]) for j in range(values.shape[1]) if i + j >= 3)),
        "BTTS Ano": float(sum(values[i, j] for i in range(1, values.shape[0]) for j in range(1, values.shape[1]))),
        "Pokrytá hmota": float(values.sum()),
    }


def no_vig_probs(odds: list[float]) -> list[float]:
    implied = np.array([1 / x for x in odds], dtype=float)
    return list(implied / implied.sum())


def ev_pct(probability: float, odds: float) -> float:
    return probability * odds - 1


def kelly_fraction(probability: float, odds: float, fraction: float) -> float:
    b = odds - 1
    full = max(0.0, (b * probability - (1 - probability)) / b)
    return full * fraction


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def init_state() -> None:
    if "ledger" not in st.session_state:
        st.session_state.ledger = pd.DataFrame(columns=[
            "created_at", "home", "away", "market", "odds_taken", "model_probability",
            "no_vig_probability", "edge", "ev_pct", "stake", "status", "result", "profit_loss"
        ])


init_state()
st.title("⚽ Football Betting Research Lab")
st.caption("Výzkumný MVP pro modelování pravděpodobností a paper betting. Bez napojení na sázkové kanceláře a bez automatického sázení.")

with st.sidebar:
    st.header("Nastavení modelu")
    shrinkage = st.slider("Stabilizace malého vzorku", 0.0, 20.0, 5.0, 0.5, help="Vyšší hodnota více přibližuje týmové statistiky ligovému průměru.")
    max_goals = st.slider("Maximum gólů v matici", 5, 12, 8)
    st.divider()
    st.header("Rizikový rámec")
    bankroll = st.number_input("Paper bankroll", min_value=100.0, value=10000.0, step=500.0)
    kelly_part = st.selectbox("Kelly frakce", [0.0, 0.25, 0.5], index=1, format_func=lambda x: "Flat stake" if x == 0 else f"{x:.0%} Kelly")
    flat_pct = st.slider("Flat stake (% banku)", 0.25, 3.0, 1.0, 0.25) / 100
    stake_cap_pct = st.slider("Max. stake (% banku)", 0.5, 5.0, 2.0, 0.5) / 100
    min_edge = st.slider("Min. modelová edge", 0.00, 0.15, 0.03, 0.005)

st.header("1. Historická data")
upload = st.file_uploader("Nahraj CSV s ukončenými zápasy", type="csv")

if upload is None:
    st.info("Nahraj vlastní CSV. Povinné sloupce: Date, HomeTeam, AwayTeam, FTHG, FTAG. Volitelně: B365H, B365D, B365A.")
    st.code("Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n2025-08-10,Team A,Team B,2,1,1.95,3.60,4.10", language="text")
    st.stop()

try:
    raw = pd.read_csv(upload)
    matches = prepare_matches(raw)
except Exception as exc:
    st.error(f"Data se nepodařilo načíst: {exc}")
    st.stop()

if len(matches) < 50:
    st.warning("Dataset obsahuje méně než 50 platných zápasů. Výsledky budou velmi nestabilní; použij více sezón.")

summary = league_summary(matches)
ratings = team_ratings(matches, shrinkage)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Platné zápasy", summary["matches"])
c2.metric("Týmy", summary["teams"])
c3.metric("Průměr domácích gólů", f"{summary['home_avg']:.2f}")
c4.metric("Průměr venkovních gólů", f"{summary['away_avg']:.2f}")

with st.expander("Kontrola a týmové ratingy"):
    st.dataframe(ratings.sort_values("home_attack", ascending=False), use_container_width=True)
    st.dataframe(matches.tail(20), use_container_width=True)

st.header("2. Predikce zápasu")
teams = sorted(ratings.index.tolist())
col_a, col_b = st.columns(2)
home = col_a.selectbox("Domácí", teams, index=0)
away_candidates = [t for t in teams if t != home]
away = col_b.selectbox("Hosté", away_candidates, index=0)

lam_h, lam_a = expected_goals(home, away, ratings, summary)
matrix = probability_matrix(lam_h, lam_a, max_goals)
probs = market_probabilities(matrix)

p1, px, p2 = probs["1"], probs["X"], probs["2"]
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("λ domácí", f"{lam_h:.2f}")
mc2.metric("λ hosté", f"{lam_a:.2f}")
mc3.metric("Nejpravděpodobnější 1X2", max({"1": p1, "X": px, "2": p2}, key={"1": p1, "X": px, "2": p2}.get))
mc4.metric("Pokrytí matice", fmt_pct(probs["Pokrytá hmota"]))

outcome_table = pd.DataFrame({
    "Trh": ["1", "X", "2", "Over 2.5", "BTTS Ano"],
    "Modelová pravděpodobnost": [p1, px, p2, probs["Over 2.5"], probs["BTTS Ano"]],
})
outcome_table["Modelová pravděpodobnost"] = outcome_table["Modelová pravděpodobnost"].map(fmt_pct)
outcome_table["Férový kurz"] = [f"{1 / p:.3f}" if p > 0 else "—" for p in [p1, px, p2, probs["Over 2.5"], probs["BTTS Ano"]]]
st.dataframe(outcome_table, use_container_width=True, hide_index=True)

with st.expander("Scoreline matrix (řádky = domácí góly, sloupce = góly hostů)"):
    st.dataframe(matrix.style.format("{:.2%}"), use_container_width=True)

st.header("3. Porovnání s trhem a paper bet")
st.caption("Zadej kurz, který jsi skutečně viděl. V1 ukládá pouze výzkumný záznam — nevykonává žádnou sázku.")

has_historical_odds = all(c in matches.columns for c in ["home_odds", "draw_odds", "away_odds"])
defaults = [2.00, 3.50, 3.75]
if has_historical_odds:
    latest = matches.dropna(subset=["home_odds", "draw_odds", "away_odds"]).tail(1)
    if not latest.empty:
        defaults = [float(latest.iloc[0][c]) for c in ["home_odds", "draw_odds", "away_odds"]]

oc1, oc2, oc3 = st.columns(3)
oh = oc1.number_input("Kurz 1", min_value=1.01, value=defaults[0], step=0.01)
od = oc2.number_input("Kurz X", min_value=1.01, value=defaults[1], step=0.01)
oa = oc3.number_input("Kurz 2", min_value=1.01, value=defaults[2], step=0.01)
market_odds = [oh, od, oa]
market_no_vig = no_vig_probs(market_odds)
market_df = pd.DataFrame({
    "Výsledek": ["1", "X", "2"],
    "Kurz": market_odds,
    "Model p": [p1, px, p2],
    "No-vig trh p": market_no_vig,
})
market_df["Edge vs no-vig"] = market_df["Model p"] - market_df["No-vig trh p"]
market_df["EV %"] = [ev_pct(p, o) for p, o in zip([p1, px, p2], market_odds)]
market_df["Férový kurz modelu"] = [1 / p for p in [p1, px, p2]]
for c in ["Model p", "No-vig trh p", "Edge vs no-vig", "EV %"]:
    market_df[c] = market_df[c].map(fmt_pct)
st.dataframe(market_df, use_container_width=True, hide_index=True)

market_label = st.selectbox("Paper-bet výběr", ["1", "X", "2"])
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
sc2.metric("Edge vs no-vig", fmt_pct(edge))
sc3.metric("EV", fmt_pct(ev))
sc4.metric("Navržený paper stake", f"{stake:,.0f}")

qualifies = edge >= min_edge and ev > 0
if qualifies:
    st.success("Kandidát splňuje zvolený práh. To není predikce jisté výhry ani důkaz dlouhodobé výhody.")
else:
    st.warning("Kandidát nesplňuje zvolený práh nebo nemá kladnou EV. Pro výzkumný režim jej nezařazuj.")

if st.button("Přidat do paper-bet ledger", type="primary"):
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
    st.success("Záznam byl přidán do paper-bet ledgeru v této relaci.")

st.header("4. Paper-bet ledger")
ledger = st.session_state.ledger.copy()
if ledger.empty:
    st.info("Zatím nemáš žádné záznamy. Přidej hypotetický tip výše.")
else:
    display = ledger.copy()
    for c in ["model_probability", "no_vig_probability", "edge", "ev_pct"]:
        display[c] = display[c].astype(float).map(fmt_pct)
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Stáhnout ledger jako CSV",
        data=ledger.to_csv(index=False).encode("utf-8"),
        file_name="paper_bet_ledger.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Důležité: tento prototyp je vzdělávací analytický nástroj. Neobchází přístupy provozovatelů, neprovádí automatické sázky a nenahrazuje právní ani finanční poradenství. Před reálným použitím testuj na časově oddělených datech, kontroluj kalibraci a pracuj pouze s částkou, jejíž ztrátu si můžeš dovolit.")
