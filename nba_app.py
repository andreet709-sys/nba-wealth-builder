import streamlit as st
import pandas as pd
import requests
import time
from io import StringIO
from datetime import datetime, timedelta
import google.generativeai as genai

# NBA API imports
from nba_api.stats.endpoints import (
    playergamelog, 
    leaguedashplayerstats, 
    commonallplayers, 
    leaguedashteamstats, 
    scoreboardv2
)
from nba_api.stats.static import players

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CourtVision AI", page_icon="🧠", layout="wide")
st.title("🧠 CourtVision AI")

# --- HELPER: DATA SCRUBBER ---
def clean_id(obj):
    """Forces any ID to a clean string."""
    try:
        return str(int(float(obj)))
    except:
        return str(obj)

# --- CONFIGURE GEMINI ---
try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    pass

# --- CSS HACKS ---
st.markdown("""
<style>
    .stAppDeployButton, [data-testid="stDecoration"] { display: none !important; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"] { visibility: hidden; height: 0%; }
</style>
""", unsafe_allow_html=True)

# --- CACHED FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_team_map_v3():
    try:
        roster = commonallplayers.CommonAllPlayers(is_only_current_season=1).get_data_frames()[0]
        return pd.Series(roster.TEAM_ABBREVIATION.values, index=roster.DISPLAY_FIRST_LAST).to_dict()
    except:
        return {}

@st.cache_data(ttl=3600)
def get_live_injuries_v3():
    url = "https://www.cbssports.com/nba/injuries/"
    headers = {"User-Agent": "Mozilla/5.0"}
    team_map = get_team_map_v3()
    try:
        response = requests.get(url, headers=headers)
        tables = pd.read_html(StringIO(response.text))
        injuries = {}
        for df in tables:
            if 'Player' in df.columns:
                for _, row in df.iterrows():
                    dirty = str(row['Player']).strip()
                    status = str(row['Injury Status'])
                    clean = dirty
                    code = "Unknown"
                    for official, team in team_map.items():
                        if official in dirty:
                            clean = official
                            code = team
                            break 
                    injuries[clean] = f"{status} ({code})"
        return injuries
    except:
        return {}

@st.cache_data(ttl=86400) 
def get_defensive_rankings_v3():
    """Fetches defensive ratings."""
    try:
        # Standard call
        teams = leaguedashteamstats.LeagueDashTeamStats(season='2025-26').get_data_frames()[0]
        teams = teams.sort_values(by='DEF_RATING', ascending=False)
        
        defense_map = {}
        for _, row in teams.iterrows():
            clean_team_id = clean_id(row['TEAM_ID'])
            defense_map[clean_team_id] = {
                'Team': row['TEAM_NAME'],
                'Rating': row['DEF_RATING']
            }
        return defense_map
    except:
        return {}

@st.cache_data(ttl=3600)
def get_todays_games_v3():
    """Finds today's games using explicit Eastern Time."""
    try:
        # 🕒 DATE DEBUG: Check yesterday, today, and tomorrow to catch the game
        # This covers timezone drifts
        dates_to_check = [
            (datetime.utcnow() - timedelta(hours=5)).strftime('%m/%d/%Y'), # Today EST
            (datetime.utcnow() + timedelta(hours=19)).strftime('%m/%d/%Y') # Tomorrow (if late night)
        ]
        
        games = {}
        
        for d in dates_to_check:
            board = scoreboardv2.ScoreboardV2(game_date=d).get_data_frames()[0]
            if not board.empty:
                for _, row in board.iterrows():
                    h = clean_id(row['HOME_TEAM_ID'])
                    v = clean_id(row['VISITOR_TEAM_ID'])
                    games[h] = v
                    games[v] = h
        
        return games
    except:
        return {}

@st.cache_data(ttl=600) 
def get_league_trends_v3():
    expected_cols = ['Player', 'Matchup', 'Season PPG', 'Last 5 PPG', 'Trend (Delta)', 'Status']
    try:
        season_stats = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', per_mode_detailed='PerGame').get_data_frames()[0]
        last5_stats = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', per_mode_detailed='PerGame', last_n_games=5).get_data_frames()[0]
        last5_stats = last5_stats[last5_stats['GP'] >= 3]

        merged = pd.merge(season_stats[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'PTS']], 
                          last5_stats[['PLAYER_ID', 'PTS']], on='PLAYER_ID', suffixes=('_Season', '_L5'))

        merged['Trend (Delta)'] = merged['PTS_L5'] - merged['PTS_Season']

        games = get_todays_games_v3()         
        defense = get_defensive_rankings_v3() 

        def analyze_matchup(row):
            my_team = clean_id(row['TEAM_ID'])
            if my_team not in games: return "No Game"
            
            opp_id = games[my_team]
            if opp_id in defense:
                opp_name = defense[opp_id]['Team']
                opp_rating = defense[opp_id]['Rating']
                if opp_rating > 116.0: return f"vs {opp_name} (🟢 Soft)"
                elif opp_rating < 112.0: return f"vs {opp_name} (🔴 Tough)"
                else: return f"vs {opp_name} (⚪ Avg)"
            return "vs Unknown"

        merged['Matchup'] = merged.apply(analyze_matchup, axis=1)
        final_df = merged.rename(columns={'PLAYER_NAME': 'Player', 'PTS_Season': 'Season PPG', 'PTS_L5': 'Last 5 PPG'})
        
        def get_status(row):
            d = row['Trend (Delta)']
            if d >= 4.0: return "🔥 Super Hot"
            elif d >= 2.0: return "🔥 Heating Up"
            elif d <= -3.0: return "❄️ Ice Cold"
            elif d <= -1.5: return "❄️ Cooling Down"
            else: return "Zap"

        final_df['Status'] = final_df.apply(get_status, axis=1)
        return final_df[expected_cols].sort_values(by='Trend (Delta)', ascending=False)

    except:
        return pd.DataFrame(columns=expected_cols)

# --- HELPER: ROBUST AI GENERATOR ---
def generate_ai_response(prompt_text):
    """Tries 1.5-Flash first, falls back to Pro on 404/Error."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model.generate_content(prompt_text).text
    except:
        try:
            model = genai.GenerativeModel('gemini-pro')
            return model.generate_content(prompt_text).text
        except Exception as e:
            return f"System Error: AI Models Unreachable. ({e})"

# --- MAIN APP LAYOUT ---
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 CourtVision IQ"])

with tab1:
    st.markdown("### *Daily Intelligence Agent*")
    
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # --- RESET BUTTON ---
        if st.button("🔄 Force Reset Data"):
            st.cache_data.clear()
            st.rerun()
            
        # --- LOAD DATA ---
        injuries = get_live_injuries_v3()
        trends = get_league_trends_v3()
        defense_debug = get_defensive_rankings_v3()
        games_debug = get_todays_games_v3()
        
        # --- DEBUG METRICS (TELL ME IF THESE ARE ZERO) ---
        c1, c2 = st.columns(2)
        c1.metric("Injuries", len(injuries))
        c2.metric("Trends", len(trends))
        c3, c4 = st.columns(2)
        c3.metric("Def Teams", len(defense_debug))
        c4.metric("Games", len(games_debug) // 2) # Divide by 2 (Home+Away)
        
        if len(defense_debug) == 0:
            st.error("❌ Defense Data Failed to Load!")
        if len(games_debug) == 0:
            st.warning("⚠️ No Games Found Today")

        st.write("---")
        st.header("🌞 Morning Briefing")
        
        with st.expander("⚠️ Impact Players OUT", expanded=False):
            found_impact = False
            if not trends.empty:
                impact_names = trends[trends['Season PPG'] > 12]['Player'].tolist()
                for star in impact_names:
                    for inj, status in injuries.items():
                        if star in inj: 
                            st.error(f"**{star}**: {status}")
                            found_impact = True
            if not found_impact: st.success("✅ No impact players out.")

    # --- MAIN TABLE ---
    st.subheader("🔥 Trends (Top Scorers)")
    if not trends.empty:
        st.dataframe(trends.head(15), hide_index=True)
    else:
        st.warning("⚠️ Market Data Unavailable.")

with tab2:
    st.header("CourtVision IQ Chat")
    if "messages" not in st.session_state: st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about matchups..."):
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing..."):
            context = f"TRENDS DATA:\n{trends.to_string()}\n\nINJURIES:\n{injuries}"
            final_prompt = f"ROLE: NBA Analyst. DATA: {context}. QUESTION: {prompt}"
            
            # Use the robust generator
            reply = generate_ai_response(final_prompt)
            
        with st.chat_message("assistant"): st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
