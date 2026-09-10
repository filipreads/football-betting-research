"""Real-time and live data source connectors"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import streamlit as st

class FootballDataConnector:
    """Fetch live match data from free/public APIs"""
    
    def __init__(self):
        # Free APIs: football-data.org, api-football.com (free tier)
        self.football_data_key = st.secrets.get("FOOTBALL_DATA_API_KEY", None)
        self.cache_ttl = 300  # 5 minutes for live data
    
    @st.cache_data(ttl=300)
    def fetch_upcoming_matches(self, league: str = "PL") -> pd.DataFrame:
        """Fetch upcoming matches from football-data.org"""
        if not self.football_data_key:
            st.warning("API key not configured. Using demo data.")
            return self._demo_upcoming()
        
        url = f"https://api.football-data.org/v4/competitions/{league}/matches"
        headers = {"X-Auth-Token": self.football_data_key}
        
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
            return pd.DataFrame(matches)
        except Exception as e:
            st.error(f"Failed to fetch live matches: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=600)
    def fetch_completed_matches(self, league: str = "PL", days_back: int = 30) -> pd.DataFrame:
        """Fetch recently completed matches"""
        if not self.football_data_key:
            return self._demo_completed()
        
        url = f"https://api.football-data.org/v4/competitions/{league}/matches"
        headers = {"X-Auth-Token": self.football_data_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            matches = []
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            for match in data.get("matches", []):
                if match["status"] == "FINISHED":
                    match_date = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))
                    if match_date > cutoff_date:
                        matches.append({
                            "date": match["utcDate"],
                            "home": match["homeTeam"]["name"],
                            "away": match["awayTeam"]["name"],
                            "home_goals": match["score"]["fullTime"]["home"],
                            "away_goals": match["score"]["fullTime"]["away"],
                            "match_id": match["id"]
                        })
            return pd.DataFrame(matches)
        except Exception as e:
            st.error(f"Failed to fetch completed matches: {e}")
            return pd.DataFrame()
    
    def _demo_upcoming(self) -> pd.DataFrame:
        """Demo data for testing without API key"""
        return pd.DataFrame([
            {"date": (datetime.utcnow() + timedelta(days=1)).isoformat(), 
             "home": "Manchester United", "away": "Liverpool", "status": "SCHEDULED", "match_id": 1},
            {"date": (datetime.utcnow() + timedelta(days=2)).isoformat(), 
             "home": "Arsenal", "away": "Chelsea", "status": "SCHEDULED", "match_id": 2},
        ])
    
    def _demo_completed(self) -> pd.DataFrame:
        """Demo historical data"""
        return pd.DataFrame([
            {"date": (datetime.utcnow() - timedelta(days=7)).isoformat(), 
             "home": "Manchester United", "away": "Liverpool", "home_goals": 2, "away_goals": 1, "match_id": 10},
            {"date": (datetime.utcnow() - timedelta(days=6)).isoformat(), 
             "home": "Arsenal", "away": "Chelsea", "home_goals": 1, "away_goals": 1, "match_id": 11},
        ])

class OddsConnector:
    """Fetch live odds from bookmakers"""
    
    def __init__(self):
        self.odds_api_key = st.secrets.get("ODDS_API_KEY", None)
    
    @st.cache_data(ttl=60)  # Very short TTL for live odds
    def fetch_live_odds(self, league: str = "soccer_england_premier_league") -> pd.DataFrame:
        """Fetch current odds from The Odds API"""
        if not self.odds_api_key:
            st.warning("Odds API key not configured. Using simulated odds.")
            return self._demo_odds()
        
        url = "https://api.the-odds-api.com/v4/sports/{}/events".format(league)
        params = {
            "apiKey": self.odds_api_key,
            "markets": "h2h,over_under",
            "oddsFormat": "decimal"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            odds_records = []
            for event in data.get("events", [])[:10]:  # Limit for demo
                if event["status"] == "scheduled":
                    for bookmaker in event.get("bookmakers", [])[:3]:  # Top 3 bookmakers
                        for market in bookmaker.get("markets", []):
                            if market["key"] == "h2h":
                                odds_records.append({
                                    "home": event["home_team"],
                                    "away": event["away_team"],
                                    "bookmaker": bookmaker["title"],
                                    "home_odds": next((o["odds"] for o in market["outcomes"] if o["name"] == event["home_team"]), None),
                                    "draw_odds": next((o["odds"] for o in market["outcomes"] if o["name"] == "Draw"), None),
                                    "away_odds": next((o["odds"] for o in market["outcomes"] if o["name"] == event["away_team"]), None),
                                })
            return pd.DataFrame(odds_records)
        except Exception as e:
            st.warning(f"Failed to fetch live odds: {e}")
            return self._demo_odds()
    
    def _demo_odds(self) -> pd.DataFrame:
        """Demo odds data"""
        return pd.DataFrame([
            {"home": "Manchester United", "away": "Liverpool", "bookmaker": "Bet365", 
             "home_odds": 2.10, "draw_odds": 3.40, "away_odds": 3.65},
            {"home": "Arsenal", "away": "Chelsea", "bookmaker": "Bet365", 
             "home_odds": 1.95, "draw_odds": 3.60, "away_odds": 4.10},
        ])
