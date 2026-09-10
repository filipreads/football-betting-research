"""Real-time and live data source connectors for Football Betting Research Platform

Supports multiple data sources:
- football-data.org API (free tier)
- The Odds API (free tier)
- Demo data for testing
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import streamlit as st


class FootballDataConnector:
    """Fetch live match data from football-data.org API"""
    
    def __init__(self):
        self.api_key = st.secrets.get("FOOTBALL_DATA_API_KEY", None)
        self.base_url = "https://api.football-data.org/v4"
        self.cache_ttl = 300  # 5 minutes for live data
    
    @st.cache_data(ttl=300)
    def fetch_upcoming_matches(self, league: str = "PL") -> pd.DataFrame:
        """Fetch upcoming matches from football-data.org
        
        Args:
            league: League code (PL=Premier League, SA=Serie A, BL1=Bundesliga, etc.)
        
        Returns:
            DataFrame with columns: date, home, away, status, match_id
        """
        if not self.api_key:
            st.warning("API key not configured. Using demo data.")
            return self._demo_upcoming()
        
        url = f"{self.base_url}/competitions/{league}/matches"
        headers = {"X-Auth-Token": self.api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            matches = []
            for match in data.get("matches", []):
                if match["status"] in ["TIMED", "SCHEDULED"]:
                    matches.append({
                        "date": match["utcDate"],
                        "home": match["homeTeam"]["name"],
                        "away": match["awayTeam"]["name"],
                        "status": match["status"],
                        "match_id": match["id"]
                    })
            
            return pd.DataFrame(matches) if matches else pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch upcoming matches: {e}")
            return self._demo_upcoming()
    
    @st.cache_data(ttl=600)
    def fetch_completed_matches(self, league: str = "PL", days_back: int = 30) -> pd.DataFrame:
        """Fetch recently completed matches
        
        Args:
            league: League code
            days_back: Number of days to look back for completed matches
        
        Returns:
            DataFrame with columns: date, home, away, home_goals, away_goals, match_id
        """
        if not self.api_key:
            return self._demo_completed()
        
        url = f"{self.base_url}/competitions/{league}/matches"
        headers = {"X-Auth-Token": self.api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            matches = []
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            for match in data.get("matches", []):
                if match["status"] == "FINISHED":
                    try:
                        match_date = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        continue
                    
                    if match_date > cutoff_date:
                        score = match.get("score", {}).get("fullTime", {})
                        home_goals = score.get("home")
                        away_goals = score.get("away")
                        
                        if home_goals is not None and away_goals is not None:
                            matches.append({
                                "date": match["utcDate"],
                                "home": match["homeTeam"]["name"],
                                "away": match["awayTeam"]["name"],
                                "home_goals": int(home_goals),
                                "away_goals": int(away_goals),
                                "match_id": match["id"]
                            })
            
            return pd.DataFrame(matches) if matches else pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch completed matches: {e}")
            return self._demo_completed()
    
    def _demo_upcoming(self) -> pd.DataFrame:
        """Demo data for testing without API key"""
        base_date = datetime.utcnow()
        return pd.DataFrame([
            {
                "date": (base_date + timedelta(days=1)).isoformat() + "Z",
                "home": "Manchester United",
                "away": "Liverpool",
                "status": "SCHEDULED",
                "match_id": 1
            },
            {
                "date": (base_date + timedelta(days=2)).isoformat() + "Z",
                "home": "Arsenal",
                "away": "Chelsea",
                "status": "SCHEDULED",
                "match_id": 2
            },
            {
                "date": (base_date + timedelta(days=3)).isoformat() + "Z",
                "home": "Manchester City",
                "away": "Tottenham",
                "status": "SCHEDULED",
                "match_id": 3
            },
        ])
    
    def _demo_completed(self) -> pd.DataFrame:
        """Demo historical data for testing"""
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
                "date": date.isoformat() + "Z",
                "home": teams_pool[home_idx],
                "away": teams_pool[away_idx],
                "home_goals": i % 4,
                "away_goals": (i + 1) % 4,
                "match_id": 100 + i
            })
        
        return pd.DataFrame(records)


class OddsConnector:
    """Fetch live odds from bookmakers via The Odds API"""
    
    def __init__(self):
        self.api_key = st.secrets.get("ODDS_API_KEY", None)
        self.base_url = "https://api.the-odds-api.com/v4"
        self.cache_ttl = 60  # Very short TTL for live odds
    
    @st.cache_data(ttl=60)
    def fetch_live_odds(self, league: str = "soccer_england_premier_league") -> pd.DataFrame:
        """Fetch current odds from The Odds API
        
        Args:
            league: League slug (e.g., 'soccer_england_premier_league')
        
        Returns:
            DataFrame with columns: home, away, bookmaker, home_odds, draw_odds, away_odds
        """
        if not self.api_key:
            st.warning("Odds API key not configured. Using simulated odds.")
            return self._demo_odds()
        
        url = f"{self.base_url}/sports/{league}/events"
        params = {
            "apiKey": self.api_key,
            "markets": "h2h",
            "oddsFormat": "decimal"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            odds_records = []
            for event in data.get("events", [])[:10]:
                if event["status"] == "scheduled":
                    for bookmaker in event.get("bookmakers", [])[:3]:
                        for market in bookmaker.get("markets", []):
                            if market["key"] == "h2h":
                                outcomes = {o["name"]: o["odds"] for o in market["outcomes"]}
                                home_odds = outcomes.get(event["home_team"])
                                away_odds = outcomes.get(event["away_team"])
                                draw_odds = outcomes.get("Draw")
                                
                                if all([home_odds, away_odds, draw_odds]):
                                    odds_records.append({
                                        "home": event["home_team"],
                                        "away": event["away_team"],
                                        "bookmaker": bookmaker["title"],
                                        "home_odds": home_odds,
                                        "draw_odds": draw_odds,
                                        "away_odds": away_odds,
                                    })
            
            return pd.DataFrame(odds_records) if odds_records else self._demo_odds()
        except Exception as e:
            st.warning(f"Failed to fetch live odds: {e}. Using demo data.")
            return self._demo_odds()
    
    def _demo_odds(self) -> pd.DataFrame:
        """Demo odds data for testing"""
        return pd.DataFrame([
            {
                "home": "Manchester United",
                "away": "Liverpool",
                "bookmaker": "Bet365",
                "home_odds": 2.10,
                "draw_odds": 3.40,
                "away_odds": 3.65,
            },
            {
                "home": "Manchester United",
                "away": "Liverpool",
                "bookmaker": "William Hill",
                "home_odds": 2.08,
                "draw_odds": 3.45,
                "away_odds": 3.70,
            },
            {
                "home": "Arsenal",
                "away": "Chelsea",
                "bookmaker": "Bet365",
                "home_odds": 1.95,
                "draw_odds": 3.60,
                "away_odds": 4.10,
            },
            {
                "home": "Manchester City",
                "away": "Tottenham",
                "bookmaker": "Bet365",
                "home_odds": 1.85,
                "draw_odds": 3.80,
                "away_odds": 4.50,
            },
        ])


class DataSourceManager:
    """Unified interface for different data sources"""
    
    def __init__(self):
        self.football = FootballDataConnector()
        self.odds = OddsConnector()
    
    def get_analysis_data(self, source: str = "demo", league: str = "PL", days_back: int = 30) -> pd.DataFrame:
        """Get match data for analysis based on selected source
        
        Args:
            source: "upload" (handled externally), "live_api", or "demo"
            league: League code
            days_back: Days back for historical data
        
        Returns:
            DataFrame ready for prepare_matches()
        """
        if source == "live_api":
            return self.football.fetch_completed_matches(league, days_back)
        elif source == "demo":
            return self.football._demo_completed()
        else:
            return pd.DataFrame()
