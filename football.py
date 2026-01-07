"""
Football Superstar Prediction Dashboard
Interactive Streamlit app to explore predictions

Author: Dale Conaghan
Version: 2.0 (Refactored)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import percentileofscore
import os
from datetime import datetime

# === CONFIG ===
CONFIG = {
    'rating_bounds': (1, 99),
    'default_stats': {'pac': 70, 'dri': 75, 'def': 25, 'phy': 70},
    'data_paths': {
        'predictions_primary': 'data/processed/final_predictions.csv',
        'predictions_fallback': 'data/processed/predictions_advanced.csv',
        'raw_2026': 'data/raw/players_data-2025_2026.csv',
        'raw_expanded': 'data/raw/players_data_expanded.csv',
        'raw_2025': 'data/raw/players_data-2024_2025.csv',
        'portfolio': 'data/processed/my_scouting_portfolio.csv'
    },
    'proprietary_metrics': ['Ghost_Factor', 'Progression_Score', 'Finishing_Alpha'],
    'dna_match_features': ['Ghost_Factor', 'Progression_Score', 'Finishing_Alpha', 'SCA90', 'Goals_per_90']
}


# === UTILITY FUNCTIONS ===

def get_base_dir():
    """Get the base directory of the application."""
    return os.path.dirname(os.path.abspath(__file__))


def safe_get(data, key, default=0):
    """Safely get a value from a dict or Series, returning default if missing or NaN."""
    if isinstance(data, dict):
        val = data.get(key, default)
    else:
        val = data.get(key, default) if hasattr(data, 'get') else data[key] if key in data else default
    return default if pd.isna(val) else val


def safe_scatter_df(df, required_cols):
    """Returns a copy with NaNs filled for Plotly scatter-safe columns."""
    df = df.copy()
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def get_pos_group(pos):
    """Standardizes position codes to readable groups."""
    if pd.isna(pos):
        return 'Unknown'
    pos = str(pos).upper()
    if 'FW' in pos:
        return 'Forward'
    if 'MF' in pos:
        return 'Midfielder'
    if 'DF' in pos:
        return 'Defender'
    if 'GK' in pos:
        return 'Goalkeeper'
    return 'Unknown'


# === DATA PROCESSING ===

def calculate_proprietary_metrics(df):
    """
    Calculate proprietary metrics: Ghost Factor, Progression Score, Finishing Alpha.
    
    All outputs are guaranteed to be NaN-free.
    """
    df = df.copy()
    
    # Ensure required columns exist and are clean
    req_cols = ['PrgC', 'PrgP', 'Gls', 'xG', 'SCA90', 'Touches']
    for c in req_cols:
        if c not in df.columns:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # 1. Progression Score (per 90)
    if '90s' in df.columns:
        nineties = pd.to_numeric(df['90s'], errors='coerce').fillna(1).replace(0, 1)
        df['Progression_Score'] = (df['PrgC'] + df['PrgP']) / nineties
    else:
        df['Progression_Score'] = df['PrgC'] + df['PrgP']
    
    # 2. Finishing Alpha (xG Overperformance)
    if 'Finishing_Alpha' not in df.columns or df['Finishing_Alpha'].sum() == 0:
        df['Finishing_Alpha'] = df['Gls'] - df['xG']
    
    # 3. Ghost Factor (Threat per Touch)
    if 'Ghost_Factor' not in df.columns or df['Ghost_Factor'].sum() == 0:
        df['Ghost_Factor'] = np.where(
            df['Touches'] > 0,
            (df['SCA90'] / df['Touches']) * 100,
            0
        )
    
    # Ensure no NaNs in output
    for col in CONFIG['proprietary_metrics']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


def apply_season_renaming(df):
    """Align column names across different season data formats."""
    df = df.copy()
    
    # Normalize rates if totals are present
    if '90s' in df.columns:
        nineties = pd.to_numeric(df['90s'], errors='coerce').fillna(1).replace(0, 1)
        rates = {'Gls': 'Goals_per_90', 'Ast': 'Assists_per_90', 'SCA': 'SCA90'}
        for col_raw, col_new in rates.items():
            if col_raw in df.columns and col_new not in df.columns:
                df[col_new] = df[col_raw] / nineties
    
    # Map existing column names
    season_rename = {
        'Gls_per_90': 'Goals_per_90',
        'Ast_per_90': 'Assists_per_90',
        'Gls.1': 'Goals_per_90',
        'Ast.1': 'Assists_per_90'
    }
    for col_old, col_new in season_rename.items():
        if col_old in df.columns and col_new not in df.columns:
            df[col_new] = df[col_old]
    
    # Final fallbacks
    if 'Goals_per_90' not in df.columns and 'Gls' in df.columns:
        df['Goals_per_90'] = df['Gls']
    if 'Assists_per_90' not in df.columns and 'Ast' in df.columns:
        df['Assists_per_90'] = df['Ast']
    
    return df


# === DATA LOADING ===

@st.cache_data
def load_data():
    """Load and prepare prediction data."""
    base_dir = get_base_dir()
    primary_path = os.path.join(base_dir, CONFIG['data_paths']['predictions_primary'])
    fallback_path = os.path.join(base_dir, CONFIG['data_paths']['predictions_fallback'])
    
    # Try primary, then fallback
    try:
        df = pd.read_csv(primary_path)
    except FileNotFoundError:
        try:
            df = pd.read_csv(fallback_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No prediction data found. Please run the pipeline first.\n"
                f"Looked for:\n  - {primary_path}\n  - {fallback_path}"
            )
    
    # Check if metrics already exist from 2026 pipeline
    has_metrics = all(
        col in df.columns and (df[col] != 0).any()
        for col in CONFIG['proprietary_metrics']
    )
    
    if has_metrics:
        return df
    
    # Fallback: Merge with raw data
    try:
        raw = load_raw_data()
        cols_to_use = ['Player', 'Squad', 'xG', 'npxG', 'SCA90', 'PrgC', 'PrgP', 'Gls', 'Touches']
        cols_to_use = [c for c in cols_to_use if c in raw.columns]
        
        df = pd.merge(df, raw[cols_to_use], on=['Player', 'Squad'], how='left', suffixes=('', '_raw'))
        df = apply_season_renaming(df)
        df = calculate_proprietary_metrics(df)
    except Exception:
        # If merge fails, just calculate what we can
        df = apply_season_renaming(df)
        df = calculate_proprietary_metrics(df)
    
    return df


@st.cache_data
def load_raw_data():
    """Load raw player data with priority for newest season."""
    base_dir = get_base_dir()
    
    paths = [
        CONFIG['data_paths']['raw_2026'],
        CONFIG['data_paths']['raw_expanded'],
        CONFIG['data_paths']['raw_2025']
    ]
    
    for path in paths:
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            return pd.read_csv(full_path)
    
    raise FileNotFoundError("No raw data files found.")


# === PORTFOLIO SYSTEM ===

def get_portfolio_path():
    """Get the path to the portfolio CSV file."""
    return os.path.join(get_base_dir(), CONFIG['data_paths']['portfolio'])


def load_portfolio():
    """Load the scouting portfolio."""
    path = get_portfolio_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=['Player', 'Club', 'DC_Index', 'Date_Scouted', 'Notes'])
    
    df = pd.read_csv(path)
    
    # Validate schema
    required_cols = ['Player', 'Club', 'DC_Index', 'Date_Scouted', 'Notes']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame(columns=required_cols)
    
    return df


def add_to_portfolio(player_row, prob_col, note):
    """Add a player to the scouting portfolio."""
    portfolio = load_portfolio()
    
    player_name = player_row['Player']
    if player_name in portfolio['Player'].values:
        return False, "Player already in portfolio."
    
    dc_index = player_row[prob_col]
    # Standardize to 0-100 if it came in as 0-1
    if dc_index <= 1.0 and dc_index > 0:
        dc_index *= 100
    
    new_entry = {
        'Player': player_name,
        'Club': player_row['Squad'],
        'DC_Index': dc_index,
        'Date_Scouted': datetime.now().strftime("%Y-%m-%d"),
        'Notes': note
    }
    
    portfolio = pd.concat([portfolio, pd.DataFrame([new_entry])], ignore_index=True)
    portfolio.to_csv(get_portfolio_path(), index=False)
    return True, f"✅ Added {player_name} to Portfolio!"


# === SIMILARITY MATCHING ===

def find_similar_players(target_name, pool_df, features):
    """
    Find players statistically similar to the target using Euclidean distance.
    
    Returns DataFrame sorted by similarity (1.0 = perfect match).
    """
    pool_df = pool_df.reset_index(drop=True).copy()
    
    # Check target exists
    target_mask = pool_df['Player'] == target_name
    if not target_mask.any():
        return None
    
    # Prepare features
    feature_data = pool_df[features].fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)
    
    # Get target vector
    target_idx = pool_df.index[target_mask][0]
    target_vector = scaled_features[target_idx].reshape(1, -1)
    
    # Calculate distances
    distances = euclidean_distances(target_vector, scaled_features)[0]
    
    # Convert to similarity (Gaussian kernel)
    sigma = np.std(distances) if np.std(distances) > 0 else 1.0
    similarity = np.exp(-distances / (2 * sigma))
    
    pool_df['Similarity'] = similarity
    
    return pool_df.sort_values('Similarity', ascending=False)


# === PLAYER CARD RENDERING ===

def render_player_card(player, prob_col, intuition=0):
    """
    Render a FIFA-style player card as HTML.
    
    Args:
        player: Dict or Series with player data
        prob_col: Column name for the main rating
        intuition: Scout override adjustment (-20 to +20)
    """
    # Convert Series to dict for safe access
    if not isinstance(player, dict):
        player = player.to_dict()
    
    # Calculate final score
    val = safe_get(player, prob_col, 50)
    if val <= 1.1:
        base_score = int(val * 99)
    else:
        base_score = int(val)
    
    min_rating, max_rating = CONFIG['rating_bounds']
    final_score = max(min_rating, min(max_rating, base_score + intuition))
    
    card_class = "platinum" if final_score >= 90 else "gold"
    
    position_group = safe_get(player, 'Position_Group', 'UNK')
    pos_display = str(position_group)[:3].upper()
    
    # Check if goalkeeper
    is_goalkeeper = 'GOA' in pos_display or 'GK' in pos_display.upper()
    
    if is_goalkeeper:
        # Goalkeeper-specific stats
        save_pct = safe_get(player, 'Save%', 70)
        div = int(min(99, 50 + (save_pct * 0.5)))  # DIV (Diving)
        
        han = int(min(99, 55 + (save_pct * 0.4)))  # HAN (Handling)
        
        cs_pct = safe_get(player, 'CS%', 30)
        kic = int(min(99, 50 + (cs_pct * 0.8)))  # KIC (Kicking)
        
        ref = int(min(99, 50 + (save_pct * 0.45)))  # REF (Reflexes)
        
        age = safe_get(player, 'Age', 25)
        spd = int(min(99, 80 - (age * 1.2)))  # SPD (Speed) - younger = faster
        
        pos = int(min(99, 55 + (save_pct * 0.35)))  # POS (Positioning)
        
        # Ensure elite GKs don't have low stats
        if final_score > 90:
            div, han, kic, ref, spd, pos = [max(s, 70) for s in [div, han, kic, ref, spd, pos]]
        
        stat_grid = f"<div>{div} DIV</div><div>{ref} REF</div><div>{han} HAN</div><div>{pos} POS</div><div>{kic} KIC</div><div>{spd} SPD</div>"
    else:
        # Outfield player stats
        prg_c = safe_get(player, 'PrgC_per_90', 0)
        pac = int(min(99, 65 + (prg_c * 5)))
        
        xg = safe_get(player, 'xG_per_90', 0)
        sho = int(min(99, 45 + (xg * 100)))
        
        sca = safe_get(player, 'SCA90', 0)
        pas = int(min(99, 55 + (sca * 8)))
        
        prog = safe_get(player, 'Progression_Score', 0)
        dri = int(min(99, 60 + (prog * 2)))
        
        tkl = safe_get(player, 'Tkl_per_90', 0)
        int_val = safe_get(player, 'Int_per_90', 0)
        def_val = int(min(99, 40 + ((tkl + int_val) * 10)))
        
        won = safe_get(player, 'Won%', 50)
        phy = int(min(99, 40 + (won * 0.7)))
        
        # Ensure elite players don't have embarrassingly low stats
        if final_score > 90:
            pac, sho, pas, dri, def_val, phy = [max(s, 60) for s in [pac, sho, pas, dri, def_val, phy]]
        
        stat_grid = f"<div>{pac} PAC</div><div>{dri} DRI</div><div>{sho} SHO</div><div>{def_val} DEF</div><div>{pas} PAS</div><div>{phy} PHY</div>"
    
    player_name = safe_get(player, 'Player', 'Unknown')
    display_name = player_name.split()[-1] if player_name else 'Unknown'
    
    est_value = safe_get(player, 'Est_Value', 0)
    
    html = f"""
    <div class="player-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="text-align: left;">
                <div class="player-rating">{final_score}</div>
                <div class="player-pos">{pos_display}</div>
            </div>
            <div style="font-size: 40px;">⚽</div>
        </div>
        <div class="player-name">{display_name}</div>
        <div class="stat-grid">
            {stat_grid}
        </div>
        <div style="margin-top: 15px; background: rgba(0,0,0,0.15); border-radius: 8px; padding: 5px; font-weight: 800; font-size: 1.3rem; border: 1px solid rgba(0,0,0,0.1);">
            <span style="font-size: 0.8rem; opacity: 0.7; display: block; margin-bottom: -5px;">EST. VALUE</span>
            £{float(est_value):.1f}M
        </div>
        <div style="margin-top: 10px; font-size: 12px; opacity: 0.8;">DC INDEX</div>
    </div>
    """
    return html


# === RADAR CHART HELPERS ===

def get_radar_config(position):
    """Get position-specific radar chart configuration."""
    configs = {
        'Goalkeeper': {
            'Shot Stopping': 'Save%',
            'Command': 'Stp%',
            'Distribution': 'Launch%',
            'Sweeping': '#OPA/90',
            'DC Index': None  # Placeholder for prob_col
        },
        'Defender': {
            'Interventions': 'Tkl+Int_per_90',
            'Aerial': 'Won%',
            'Passing': 'Cmp%',
            'Progression': 'Progression_Score',
            'DC Index': None
        },
        'Midfielder': {
            'Engine': 'Progression_Score',
            'Security': 'Cmp%',
            'Creation': 'SCA90',
            'Defense': 'Tkl+Int_per_90',
            'DC Index': None
        }
    }
    # Default for Forwards
    return configs.get(position, {
        'Scoring': 'Goals_per_90',
        'Creation': 'SCA90',
        'Intelligence': 'Ghost_Factor',
        'Engine': 'Progression_Score',
        'DC Index': None
    })


def get_radar_values(player_data, df_pool, cols_map, prob_col):
    """Calculate percentile values for radar chart."""
    vals = []
    for label, col in cols_map.items():
        actual_col = prob_col if col is None else col
        
        if actual_col not in df_pool.columns:
            vals.append(50)
            continue
        
        if isinstance(player_data, dict):
            raw_val = player_data.get(actual_col, 0)
        else:
            raw_val = player_data[actual_col] if actual_col in player_data else 0
        
        raw_val = 0 if pd.isna(raw_val) else raw_val
        
        col_data = df_pool[actual_col].dropna()
        if len(col_data) > 0:
            pct = percentileofscore(col_data, raw_val, kind='rank')
        else:
            pct = 50
        
        vals.append(pct)
    
    return vals


# === CSS STYLES ===

def get_app_css():
    """Return the CSS styles for the application."""
    return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Teko:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #e0e0e0;
            font-family: 'Inter', sans-serif;
        }
        
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 800;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #2962ff !important;
        }
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] span {
            color: #1a1a1a !important;
        }
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        /* Metric Styling */
        div[data-testid="stMetric"] {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        div[data-testid="stMetric"] label, 
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"] > div,
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"] p {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 16px !important;
            opacity: 1 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #4ade80;
            font-family: 'Teko', sans-serif;
            font-size: 36px;
        }

        /* Tab Styling */
        button[data-baseweb="tab"] {
            color: rgba(255, 255, 255, 0.7) !important;
            font-size: 18px !important;
            font-weight: 600 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #ffffff !important;
            background-color: rgba(255, 255, 255, 0.1) !important;
            border-top: 3px solid #4ade80 !important;
            border-bottom: none !important;
        }
        button[data-baseweb="tab"]:hover {
            color: #ffffff !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }
        
        /* FIFA Card Style */
        .player-card {
            width: 100%;
            max-width: 300px;
            margin: auto;
            border-radius: 20px;
            padding: 20px;
            text-align: center;
            color: #1a1a1a;
            font-family: 'Teko', sans-serif;
            box-shadow: 0 10px 20px rgba(0,0,0,0.4);
            transition: transform 0.3s ease;
        }
        .player-card:hover {
            transform: translateY(-5px);
        }
        
        .gold {
            background: linear-gradient(135deg, #e6b980 0%, #eacda3 100%);
        }
        .platinum {
            background: linear-gradient(0deg, #d3d3d3 0%, #ffffff 74%);
            border: 2px solid #aef;
            box-shadow: 0 0 15px rgba(100, 200, 255, 0.5);
        }
        
        .player-rating {
            font-size: 56px;
            font-weight: 700;
            line-height: 0.9;
        }
        .player-pos {
            font-size: 20px;
            font-weight: 600;
        }
        .player-name {
            font-size: 32px;
            font-weight: 700;
            margin: 10px 0;
            text-transform: uppercase;
            border-bottom: 2px solid rgba(0,0,0,0.1);
            letter-spacing: 1px;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            font-size: 20px;
            font-weight: 600;
        }

        /* Dropdown Visibility */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="select"] span {
            color: #000000 !important;
        }

        /* Checkbox Label - match caption style */
        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] label p,
        div[data-testid="stCheckbox"] label span,
        .stCheckbox label,
        .stCheckbox label p,
        .stCheckbox > label > div > p {
            color: rgba(255, 255, 255, 0.6) !important;
            font-size: 0.875rem !important;
            font-weight: 400 !important;
        }

        /* Comparison Names */
        .comp-name {
            color: #4ade80 !important;
            font-size: 1.6rem !important;
            font-weight: 900 !important;
            text-transform: uppercase;
            text-shadow: 0 0 10px rgba(74, 222, 128, 0.4);
        }

        /* Decision Matrix */
        .decision-matrix {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
            color: #ffffff;
        }
        .decision-matrix th {
            background: #2962ff;
            color: white;
            padding: 12px;
            text-align: left;
            font-size: 1.1rem;
        }
        .decision-matrix td {
            padding: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            font-size: 1.0rem;
        }
        .decision-matrix tr:hover {
            background: rgba(255,255,255,0.05);
        }
        .highlight-val {
            color: #4ade80;
            font-weight: 800;
        }
        </style>
    """


# === MAIN APPLICATION ===

def main():
    """Main application entry point."""
    st.set_page_config(page_title="DC Pro", page_icon="⚽", layout="wide")
    st.markdown(get_app_css(), unsafe_allow_html=True)
    
    st.title("DC PRO")
    st.markdown("### <span style='color: #4ade80'>Next-Gen Scouting Intelligence</span>", unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run the pipeline first to generate predictions.")
        return
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return
    
    # Ensure required columns exist
    for col in ['Goals_per_90', 'Assists_per_90']:
        if col not in df.columns:
            df[col] = 0
    
    # Determine probability column
    prob_col = None
    for col in ['DC_Index', 'Ensemble_Probability', 'XGB_Probability', 'DC_Score']:
        if col in df.columns:
            prob_col = col
            break
    
    if prob_col is None:
        prob_cols = [c for c in df.columns if 'Probability' in c or 'Index' in c or 'Score' in c]
        prob_col = prob_cols[0] if prob_cols else df.columns[-1]
    
    # Determine if already scaled (0-100 vs 0-1)
    is_scaled = df[prob_col].max() > 1.1 if not df.empty else True
    
    # === SIDEBAR ===
    st.sidebar.markdown("<h2 style='color: #2962ff; border-bottom: 2px solid #2962ff; padding-bottom: 5px;'>FILTERS</h2>", unsafe_allow_html=True)
    
    # Position filter
    positions = ['All'] + sorted(df['Position_Group'].dropna().unique().tolist())
    selected_pos = st.sidebar.selectbox("Position", positions)
    
    # Age filter
    min_age_data = int(df['Age'].min()) if not df.empty else 16
    min_age = min(16, min_age_data)
    max_age = int(df['Age'].max()) if not df.empty else 24
    age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))
    
    # Market value filter
    if 'Est_Value' in df.columns:
        max_v = float(df['Est_Value'].max())
        val_range = st.sidebar.slider("Max Market Value (£M)", 0.0, max_v, max_v)
    else:
        val_range = 1000.0
    
    # Probability filter
    min_prob = st.sidebar.slider("Minimum Superstar Probability", 0, 100, 50) / 100
    
    # Scout override
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h2 style='color: #2962ff; border-bottom: 2px solid #2962ff; padding-bottom: 5px;'>SCOUT OVERRIDE</h2>", unsafe_allow_html=True)
    intuition_score = st.sidebar.slider("Scout's Eye Test (Adjustment)", -20, 20, 0, help="Adjust the model score based on your domain knowledge.")
    
    # Portfolio quick add
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h2 style='color: #2962ff; border-bottom: 2px solid #2962ff; padding-bottom: 5px;'>SCOUT NOTEBOOK</h2>", unsafe_allow_html=True)
    all_players = df['Player'].sort_values().unique()
    scout_target = st.sidebar.selectbox("Select Player to Shortlist", all_players)
    scout_note = st.sidebar.text_input("Scouting Note", placeholder="e.g. 'Beast in transition'")
    
    if st.sidebar.button("Add to Portfolio"):
        target_row = df[df['Player'] == scout_target].iloc[0]
        success, msg = add_to_portfolio(target_row, prob_col, scout_note)
        if success:
            st.sidebar.success(msg)
        else:
            st.sidebar.warning(msg)
    
    # === APPLY FILTERS ===
    filtered_df = df.copy()
    
    if selected_pos != 'All':
        filtered_df = filtered_df[filtered_df['Position_Group'] == selected_pos]
    
    filtered_df = filtered_df[
        (filtered_df['Age'] >= age_range[0]) &
        (filtered_df['Age'] <= age_range[1])
    ]
    
    if 'Est_Value' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Est_Value'] <= val_range]
    
    # Probability filter
    prob_threshold = min_prob * 100 if is_scaled else min_prob
    filtered_df = filtered_df[filtered_df[prob_col] >= prob_threshold]
    
    # Calculate Value Score for ROI tab
    if 'Est_Value' in filtered_df.columns and prob_col in filtered_df.columns:
        filtered_df['Est_Value_Adj'] = filtered_df['Est_Value'].replace(0, 0.01)
        dc_scaled = filtered_df[prob_col] * 100 if not is_scaled else filtered_df[prob_col]
        filtered_df['Value_Score'] = dc_scaled * (100 / (100 + filtered_df['Est_Value_Adj']))
    
    # Ensure Tkl+Int_per_90 exists for radar charts
    if 'Tkl+Int_per_90' not in filtered_df.columns:
        tkl = filtered_df['Tkl_per_90'] if 'Tkl_per_90' in filtered_df.columns else 0
        int_val = filtered_df['Int_per_90'] if 'Int_per_90' in filtered_df.columns else 0
        filtered_df['Tkl+Int_per_90'] = pd.Series(tkl).fillna(0) + pd.Series(int_val).fillna(0)
    
    # Empty state check
    if filtered_df.empty:
        st.warning("No players match the current filters. Try adjusting the sidebar settings.")
        return
    
    # === TABS ===
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "TOP PROSPECTS", "ANALYTICS", "VALUATION / ROI", 
        "H2H ANALYSIS", "RAW DATA", "PORTFOLIO", "DNA MATCHER", "METHODOLOGY"
    ])
    
    # --- TAB 1: TOP PROSPECTS ---
    with tab1:
        render_top_prospects_tab(filtered_df, prob_col, is_scaled, intuition_score)
    
    # --- TAB 2: ANALYTICS ---
    with tab2:
        render_analytics_tab(filtered_df, prob_col)
    
    # --- TAB 3: VALUATION / ROI ---
    with tab3:
        render_valuation_tab(filtered_df, prob_col)
    
    # --- TAB 4: H2H ANALYSIS ---
    with tab4:
        render_h2h_tab(filtered_df, prob_col, intuition_score, is_scaled)
    
    # --- TAB 5: RAW DATA ---
    with tab5:
        st.header("Raw Intelligence Data")
        
        # Clean up the dataframe for display
        display_df = filtered_df.sort_values(prob_col, ascending=False).copy()
        
        # Clean Nation column if it exists
        if 'Nation' in display_df.columns:
            # Map country codes to full names
            country_map = {
                'ENG': 'ENGLAND', 'ESP': 'SPAIN', 'FRA': 'FRANCE', 'GER': 'GERMANY',
                'ITA': 'ITALY', 'POR': 'PORTUGAL', 'NED': 'NETHERLANDS', 'BEL': 'BELGIUM',
                'BRA': 'BRAZIL', 'ARG': 'ARGENTINA', 'URU': 'URUGUAY', 'COL': 'COLOMBIA',
                'MEX': 'MEXICO', 'USA': 'USA', 'CAN': 'CANADA', 'JPN': 'JAPAN',
                'KOR': 'SOUTH KOREA', 'AUS': 'AUSTRALIA', 'NGA': 'NIGERIA', 'GHA': 'GHANA',
                'SEN': 'SENEGAL', 'CIV': 'IVORY COAST', 'CMR': 'CAMEROON', 'MAR': 'MOROCCO',
                'EGY': 'EGYPT', 'ALG': 'ALGERIA', 'TUN': 'TUNISIA', 'RSA': 'SOUTH AFRICA',
                'WAL': 'WALES', 'SCO': 'SCOTLAND', 'IRL': 'IRELAND', 'NIR': 'N. IRELAND',
                'AUT': 'AUSTRIA', 'SUI': 'SWITZERLAND', 'POL': 'POLAND', 'CZE': 'CZECHIA',
                'CRO': 'CROATIA', 'SRB': 'SERBIA', 'UKR': 'UKRAINE', 'RUS': 'RUSSIA',
                'TUR': 'TURKEY', 'GRE': 'GREECE', 'DEN': 'DENMARK', 'SWE': 'SWEDEN',
                'NOR': 'NORWAY', 'FIN': 'FINLAND', 'CHI': 'CHILE', 'PAR': 'PARAGUAY',
                'ECU': 'ECUADOR', 'PER': 'PERU', 'VEN': 'VENEZUELA', 'BOL': 'BOLIVIA',
                'JAM': 'JAMAICA', 'HON': 'HONDURAS', 'CRC': 'COSTA RICA', 'PAN': 'PANAMA',
                'SVN': 'SLOVENIA', 'SVK': 'SLOVAKIA', 'HUN': 'HUNGARY', 'ROU': 'ROMANIA',
                'BUL': 'BULGARIA', 'ALB': 'ALBANIA', 'MKD': 'N. MACEDONIA', 'BIH': 'BOSNIA',
                'MNE': 'MONTENEGRO', 'KOS': 'KOSOVO', 'GEO': 'GEORGIA', 'ARM': 'ARMENIA',
                'ISR': 'ISRAEL', 'IRN': 'IRAN', 'IRQ': 'IRAQ', 'QAT': 'QATAR',
                'KSA': 'SAUDI ARABIA', 'UAE': 'UAE', 'CHN': 'CHINA', 'IND': 'INDIA',
            }
            
            def clean_nation(val):
                if pd.isna(val):
                    return ''
                # Extract the 3-letter code (last 3 chars or after space)
                val_str = str(val).strip()
                if ' ' in val_str:
                    code = val_str.split()[-1].upper()
                else:
                    code = val_str.upper()
                return country_map.get(code, code)
            
            display_df['Nation'] = display_df['Nation'].apply(clean_nation)
        
        st.dataframe(display_df, use_container_width=True)
    
    # --- TAB 6: PORTFOLIO ---
    with tab6:
        render_portfolio_tab(is_scaled)
    
    # --- TAB 7: DNA MATCHER ---
    with tab7:
        render_dna_matcher_tab(filtered_df, prob_col, intuition_score)
    
    # --- TAB 8: METHODOLOGY ---
    with tab8:
        render_methodology_tab()


def render_methodology_tab():
    """Render the Methodology tab explaining how the DC Index works."""
    st.header("Methodology")
    st.markdown("*How the DC Index identifies future superstars*")
    
    st.markdown("---")
    
    # === DATA SOURCES ===
    st.subheader("1. Data Sources")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Primary Source:** FBRef.com (via StatsBomb)
        
        **Coverage:**
        - Big 5 European Leagues
        - Premier League, La Liga, Bundesliga, Serie A, Ligue 1
        - Seasons: 2021-22 through 2025-26
        """)
    with col2:
        st.metric("Total Player-Seasons", "14,229")
        st.metric("Training Sample (U23, 500+ mins)", "1,599")
        st.metric("Known Breakout Players", "56")
    
    st.markdown("---")
    
    # === DC INDEX METHODOLOGY ===
    st.subheader("2. DC Index: Empirical Weight Training")
    
    st.markdown("""
    The DC Index uses **logistic regression** to identify which U23 metrics best predict future superstar status.
    
    **Training Process:**
    1. Filtered historical data to U23 players with 500+ minutes
    2. Labelled "breakout" players (Bellingham, Saka, Musiala, Pedri, etc.)
    3. Trained position-specific models to find predictive features
    4. Extracted coefficients as empirical weights
    """)
    
    # Learned weights
    st.markdown("### Learned Weights by Position")
    
    weights_col1, weights_col2, weights_col3 = st.columns(3)
    
    with weights_col1:
        st.markdown("**Forwards**")
        st.code("""
PrgC_per_90:  +2.21  ⬆️
xAG_per_90:   +1.90  ⬆️
xG_per_90:    +1.73  ⬆️
Age:          -0.56  ⬇️
        """)
    
    with weights_col2:
        st.markdown("**Midfielders**")
        st.code("""
PrgP_per_90:  +1.19  ⬆️
PrgC_per_90:  +0.82  ⬆️
KP_per_90:    +0.61  ⬆️
Age:          -0.37  ⬇️
        """)
    
    with weights_col3:
        st.markdown("**Defenders**")
        st.code("""
PrgP_per_90:  +2.00  ⬆️
Tkl_per_90:   +1.52  ⬆️
Age:          -0.52  ⬇️
PrgC_per_90:  +0.11  ⬆️
        """)
    
    st.info("**Key Finding:** Progressive ball movement (PrgC, PrgP) is the strongest predictor of breakout potential across all positions.")
    
    st.markdown("---")
    
    # === VALUATION MODEL ===
    st.subheader("3. Market Valuation Model")
    
    st.markdown("""
    Estimated market values are calculated using six factors:
    """)
    
    val_col1, val_col2 = st.columns(2)
    
    with val_col1:
        st.markdown("""
        | Factor | Description |
        |--------|-------------|
        | **DC Index** | Base potential score |
        | **Age** | U19: +50%, U22: +20%, 25+: -20% |
        | **League** | PL: +50%, Top 4: +20%, Ligue 1: +10% |
        """)
    
    with val_col2:
        st.markdown("""
        | Factor | Description |
        |--------|-------------|
        | **Position** | FW: +25%, MID: +10%, DEF: -10%, GK: -30% |
        | **Minutes** | 2500+: +10%, <500: -40% |
        | **Form** | Outperforming xG: up to +15% |
        """)
    
    st.markdown("---")
    
    # === PROPRIETARY METRICS ===
    st.subheader("4. Proprietary Metrics")
    
    met_col1, met_col2, met_col3 = st.columns(3)
    
    with met_col1:
        st.markdown("**Finishing Alpha**")
        st.markdown("Goals minus Expected Goals (xG). Measures clinical finishing ability.")
        st.latex(r"\alpha = Goals - xG")
    
    with met_col2:
        st.markdown("**Ghost Factor**")
        st.markdown("Threat created per touch. Identifies efficient, low-touch danger players.")
        st.latex(r"Ghost = \frac{SCA}{Touches} \times 100")
    
    with met_col3:
        st.markdown("**Progression Score**")
        st.markdown("Combined progressive carries and passes per 90. Measures ball advancement.")
        st.latex(r"Prog = \frac{PrgC + PrgP}{90s}")
    
    st.markdown("---")
    
    # === LIMITATIONS ===
    st.subheader("5. Limitations & Future Work")
    
    st.markdown("""
    **Current Limitations:**
    - No contract length data (affects real transfer fees significantly)
    - Limited goalkeeper metrics in source data
    - No injury history or fitness data
    - Market valuations are heuristic, not trained on real fees
    
    **Future Improvements:**
    - Scrape Transfermarkt for actual transfer fees to train valuation model
    - Add contract expiry data
    - Incorporate trajectory (season-over-season improvement)
    - Expand to more leagues (Eredivisie, Liga Portugal, etc.)
    """)
    
    st.markdown("---")
    
    # === CITATION ===
    st.subheader("6. Citation")
    st.code("""
Conaghan, D. (2026). DC Index: Empirical Football Talent Identification.

Data: FBRef.com / StatsBomb
    """, language=None)


# === TAB RENDER FUNCTIONS ===

def render_top_prospects_tab(filtered_df, prob_col, is_scaled, intuition_score):
    """Render the Top Prospects tab."""
    st.header("The DC Index: Top Prospects")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prospects in View", len(filtered_df))
    with col2:
        avg_score = filtered_df[prob_col].mean()
        display_score = avg_score if is_scaled else avg_score * 100
        st.metric("Avg DC Score", f"{display_score:.1f}")
    with col3:
        threshold = 90 if is_scaled else 0.9
        high_prob = (filtered_df[prob_col] >= threshold).sum()
        st.metric("Elite Tier (>90)", high_prob)
    with col4:
        avg_age = filtered_df['Age'].mean()
        st.metric("Avg Age", f"{avg_age:.1f}")
    
    # Top 3 Cards
    st.subheader("Market Leaders")
    # Sort by DC Index, then by Est_Value as tiebreaker
    top3 = filtered_df.sort_values([prob_col, 'Est_Value'], ascending=[False, False]).head(3)
    
    c1, c2, c3 = st.columns(3)
    for i, (idx, row) in enumerate(top3.iterrows()):
        with [c1, c2, c3][i]:
            st.markdown(render_player_card(row, prob_col, intuition_score), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top 20 table
    display_cols = ['Player', 'Age', 'Squad', 'Position_Group', 'Goals_per_90', 'Assists_per_90', prob_col]
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    
    # Sort by DC Index, then by Est_Value as tiebreaker
    top20 = filtered_df.sort_values([prob_col, 'Est_Value'], ascending=[False, False]).head(20)[available_cols].copy()
    top20.columns = ['Player', 'Age', 'Club', 'Position', 'Goals/90', 'Assists/90', 'DC Index'][:len(available_cols)]
    
    format_dict = {
        'DC Index': '{:.1f}' if is_scaled else '{:.1%}',
        'Goals/90': '{:.2f}',
        'Assists/90': '{:.2f}'
    }
    format_dict = {k: v for k, v in format_dict.items() if k in top20.columns}
    
    st.dataframe(
        top20.style.background_gradient(subset=['DC Index'], cmap='Greens').format(format_dict),
        use_container_width=True,
        hide_index=True
    )


def render_analytics_tab(filtered_df, prob_col):
    """Render the Analytics tab."""
    st.header("Analytical Deep Dive")
    st.markdown("<div style='color: rgba(255,255,255,0.7); margin-bottom: 20px; font-style: italic;'>Advanced metrics visualization powered by Plotly.</div>", unsafe_allow_html=True)
    
    # Proprietary metrics summary
    st.markdown("### Proprietary 'Secret' Intelligence")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        top_finisher = filtered_df.sort_values('Finishing_Alpha', ascending=False).iloc[0]
        st.metric(
            "Top Finisher (Alpha)",
            top_finisher['Player'],
            f"+{filtered_df['Finishing_Alpha'].max():.2f} xG"
        )
    with c2:
        top_prog = filtered_df.sort_values('Progression_Score', ascending=False).iloc[0]
        st.metric(
            "Most Progressive",
            top_prog['Player'],
            f"{filtered_df['Progression_Score'].max():.0f} Acts"
        )
    with c3:
        top_ghost = filtered_df.sort_values('Ghost_Factor', ascending=False).iloc[0]
        st.metric(
            "Ghost Factor (Efficiency)",
            top_ghost['Player'],
            f"{filtered_df['Ghost_Factor'].max():.2f}"
        )
    
    st.markdown("---")
    
    # Scatter plots (with NaN-safe data)
    plot_cols = [prob_col, 'Finishing_Alpha', 'Progression_Score', 'Ghost_Factor', 'SCA90', 'Touches', 'Goals_per_90', 'Assists_per_90']
    plot_df = safe_scatter_df(filtered_df, plot_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("System vs Killer?")
        with st.expander("How to read this"):
            st.markdown("""
            * **Top Right:** High Index + High Finishing Alpha = **Lethal Finishers**
            * **Bottom Right:** High Index + Negative Alpha = **Underperformers**
            """)
        fig = px.scatter(
            plot_df, x=prob_col, y='Finishing_Alpha',
            color='Position_Group', size='Progression_Score', hover_name='Player',
            title='Value Finding: High Index + High Finishing Alpha', template='plotly_dark'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Underperforming xG")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.caption("The Haaland Scale")
        with st.expander("How to read this"):
            st.markdown("""
            * **Top Left (The Ghost):** Low Touches + High Danger. Efficient assassins.
            * **Bottom Right (Passenger):** High Touches + Low Danger. Stat-padders.
            """)
        fig = px.scatter(
            plot_df, x='Touches', y='SCA90',
            color='Ghost_Factor', size=prob_col, hover_name='Player',
            title='The "Ghost" Chart: High Danger, Low Touches', template='plotly_dark',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("Bias Check")
        with st.expander("How to read this"):
            st.markdown("""If Forwards are much higher than Defenders, mentally adjust expectations.
            
**Note:** DC Index is calculated *within* each position group — a 90-rated goalkeeper is in the top 10% of goalkeepers, not compared to forwards. This ensures fair comparison across positions.""")
        pos_avg = plot_df.groupby('Position_Group')[prob_col].mean().reset_index()
        fig = px.bar(pos_avg, x='Position_Group', y=prob_col,
                     title='Index by Position', color=prob_col, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.caption("Hidden Gem Finder")
        with st.expander("How to read this"):
            st.markdown("**Look for Bottom-Left / Dark Green:** LOW output but HIGH DC Index = hidden potential.")
        fig = px.scatter(
            plot_df, x='Goals_per_90', y='Assists_per_90',
            color=prob_col, size=prob_col, hover_name='Player',
            title='Goal Contributions vs Index', template='plotly_dark',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_valuation_tab(filtered_df, prob_col):
    """Render the Valuation/ROI tab."""
    st.header("Financial Intelligence (ROI Analysis)")
    st.markdown("""
    <div style='background: rgba(41, 98, 255, 0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #2962ff; margin-bottom: 25px;'>
        <b>ROI INDEX:</b> Identifies players with <b>Elite Stats</b> who are currently <b>undervalued</b>.
    </div>
    """, unsafe_allow_html=True)
    
    v1, v2, v3 = st.columns(3)
    
    if 'Value_Score' in filtered_df.columns and 'Est_Value' in filtered_df.columns:
        # Find best bargain (quality > 50)
        potential_bargains = filtered_df[filtered_df[prob_col] >= 50]
        if not potential_bargains.empty:
            bargain = potential_bargains.sort_values('Value_Score', ascending=False).iloc[0]
        else:
            bargain = filtered_df.sort_values('Value_Score', ascending=False).iloc[0]
        
        v1.metric("Ultimate Bargain", bargain['Player'], f"ROI: {bargain['Value_Score']:.2f}")
        
        expensive = filtered_df.sort_values('Est_Value', ascending=False).iloc[0]
        v2.metric("Market Leader", expensive['Player'], f"£{expensive['Est_Value']:.0f}M")
        
        avg_roi = filtered_df['Value_Score'].mean()
        v3.metric("Avg. Market ROI", f"{avg_roi:.2f}", "Efficiency Score")
        
        st.markdown("---")
        
        # ROI scatter plot
        plot_df = safe_scatter_df(filtered_df, [prob_col, 'Est_Value', 'Value_Score'])
        fig = px.scatter(
            plot_df, x='Est_Value', y=prob_col,
            color='Value_Score', size='Value_Score', hover_name='Player',
            labels={'Est_Value': 'Market Value (£M)', prob_col: 'DC Index'},
            title='The Value Curve: DC Index vs Market Price', template='plotly_dark',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("High-ROI Shortlist")
        roi_table = filtered_df.sort_values('Value_Score', ascending=False).head(10)[
            ['Player', 'Squad', 'Age', 'Est_Value', 'Value_Score']
        ].copy()
        roi_table['Age'] = roi_table['Age'].astype(int)
        st.dataframe(
            roi_table.style.background_gradient(subset=['Value_Score'], cmap='Greens')
                     .format({'Est_Value': '£{:.1f}M', 'Value_Score': '{:.2f}'}),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("⚠️ Market valuation data not found.")
        st.info("💡 **TIP:** Clear cache (hamburger menu → Clear Cache) to load new data.")


def render_h2h_tab(filtered_df, prob_col, intuition_score, is_scaled):
    """Render the Head-to-Head Analysis tab."""
    st.header("Head-to-Head Analysis")
    
    players = filtered_df['Player'].tolist()
    if len(players) < 2:
        st.warning("Need at least 2 players for comparison.")
        return
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<p class='comp-name'>SELECT TARGET</p>", unsafe_allow_html=True)
        p1_name = st.selectbox("Select Target", players, index=0, label_visibility="collapsed")
    with c2:
        st.markdown("<p class='comp-name'>SELECT BENCHMARK</p>", unsafe_allow_html=True)
        p2_name = st.selectbox("Select Benchmark", players, index=min(1, len(players)-1), label_visibility="collapsed")
    
    if not p1_name or not p2_name:
        return
    
    # Convert to dicts for safe mutation
    p1 = filtered_df[filtered_df['Player'] == p1_name].iloc[0].to_dict()
    p2 = filtered_df[filtered_df['Player'] == p2_name].iloc[0].to_dict()
    
    # Player cards
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(render_player_card(p1, prob_col, intuition_score), unsafe_allow_html=True)
    with c2:
        st.markdown(render_player_card(p2, prob_col, intuition_score), unsafe_allow_html=True)
    
    # Get position-specific radar config
    pos_1 = p1.get('Position_Group', 'Forward')
    radar_cols = get_radar_config(pos_1)
    
    # Calculate radar values
    p1_vals = get_radar_values(p1, filtered_df, radar_cols, prob_col)
    p2_vals = get_radar_values(p2, filtered_df, radar_cols, prob_col)
    
    # Similarity calculation
    overlap = 100 - np.mean(np.abs(np.array(p1_vals) - np.array(p2_vals)))
    
    st.markdown(f"""
        <div style='text-align: center; background: rgba(74, 222, 128, 0.1); padding: 10px; border-radius: 10px; border: 1px solid #4ade80;'>
            <h3 style='margin: 0; color: #4ade80;'>Tactical Similarity: {overlap:.1f}%</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Radar chart
    categories = list(radar_cols.keys())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=p1_vals + [p1_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=p1_name,
        line_color='#4ade80',
        fillcolor='rgba(74, 222, 128, 0.3)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=p2_vals + [p2_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=p2_name,
        line_color='#ffffff',
        fillcolor='rgba(255, 255, 255, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        title=dict(text="High-Resolution Profile Comparison", font=dict(size=28, color='#ffffff'), x=0.5),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(size=22, color='#ffffff'))
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Decision matrix
    st.markdown("### Position-Specific Decision Matrix")
    
    rows_html = ""
    for label, col in radar_cols.items():
        actual_col = prob_col if col is None else col
        val1 = p1.get(actual_col, 0) or 0
        val2 = p2.get(actual_col, 0) or 0
        
        if '%' in str(actual_col) or 'Percent' in label:
            f1, f2 = f"{val1:.1f}%", f"{val2:.1f}%"
        else:
            f1, f2 = f"{val1:.2f}", f"{val2:.2f}"
        
        rows_html += f"<tr><td>{label}</td><td>{f1}</td><td>{f2}</td></tr>"
    
    matrix_html = f"""
    <table class='decision-matrix'>
        <thead>
            <tr>
                <th>Tactical Metric ({pos_1} Profile)</th>
                <th>{p1_name}</th>
                <th>{p2_name}</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Match Similarity</td><td class='highlight-val'>100% (TARGET)</td><td class='highlight-val'>{overlap:.1f}%</td></tr>
            <tr><td>Current Club</td><td>{p1.get('Squad', 'N/A')}</td><td>{p2.get('Squad', 'N/A')}</td></tr>
            <tr><td>Age</td><td>{int(p1.get('Age', 0))}</td><td>{int(p2.get('Age', 0))}</td></tr>
            <tr><td>Market Value</td><td>£{float(p1.get('Est_Value', 0)):.1f}M</td><td>£{float(p2.get('Est_Value', 0)):.1f}M</td></tr>
            <tr style='background: rgba(41, 98, 255, 0.1);'><td colspan='3' style='text-align: center; font-weight: 800; font-size: 0.8rem; letter-spacing: 2px; color: #2962ff;'>TACTICAL DNA</td></tr>
            {rows_html}
        </tbody>
    </table>
    """
    st.markdown(matrix_html, unsafe_allow_html=True)


def render_portfolio_tab(is_scaled):
    """Render the Portfolio tab."""
    st.header("My Scouting Portfolio")
    portfolio_df = load_portfolio()
    
    if portfolio_df.empty:
        st.info("Your portfolio is empty. Go to the Sidebar to add players!")
        return
    
    # Metrics
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Shortlisted", len(portfolio_df))
    with c2:
        scores = pd.to_numeric(portfolio_df['DC_Index'], errors='coerce')
        avg_val = scores.mean()
        is_p_scaled = scores.max() > 1.1
        display_p = avg_val if is_p_scaled else avg_val * 100
        st.metric("Portfolio Quality (Avg)", f"{display_p:.1f}")
    
    # Display
    st.dataframe(
        portfolio_df.style.background_gradient(subset=['DC_Index'], cmap='Greens')
                    .format({'DC_Index': '{:.1f}' if is_scaled else '{:.1%}'}),
        use_container_width=True,
        hide_index=True
    )
    
    # Download
    csv = portfolio_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 EXPORT REPORT",
        csv,
        "dc_scout_portfolio.csv",
        "text/csv",
        key='download-csv',
        type='primary',
        use_container_width=True
    )


def render_dna_matcher_tab(filtered_df, prob_col, intuition_score):
    """Render the DNA Matcher tab."""
    st.header("DNA Matcher: Find Replacements")
    st.markdown("> **Scenario:** You lost your star. Find a clone.")
    
    # Load full raw data
    try:
        raw_all = load_raw_data()
        raw_all = apply_season_renaming(raw_all)
        raw_all = calculate_proprietary_metrics(raw_all)
        
        if 'Position_Group' not in raw_all.columns:
            if 'Pos' in raw_all.columns:
                raw_all['Position_Group'] = raw_all['Pos'].apply(get_pos_group)
            else:
                raw_all['Position_Group'] = 'Unknown'
        
        all_players_search = sorted(raw_all['Player'].unique().tolist())
    except Exception:
        raw_all = filtered_df.copy()
        all_players_search = sorted(filtered_df['Player'].unique().tolist())
    
    if not all_players_search:
        st.error("No players available for search.")
        return
    
    st.markdown("---")
    st.markdown("### Step 1: Select the Star to Replace")
    st.caption("Search any player (even if over 23). We'll find their younger clone.")
    
    target_name = st.selectbox("Search Player Database:", all_players_search, index=0, label_visibility="collapsed")
    
    # Position matching toggle
    st.markdown("### Step 2: Configure Matching")
    match_pos = st.checkbox("Strict Position Matching", value=True)
    st.caption("**ON** = direct clones only. **OFF** = skillset twins across positions.")
    
    st.markdown("---")
    st.markdown("### Step 3: Run the Algorithm")
    
    if st.button("SCAN DATABASE FOR MATCHES", type="primary", use_container_width=True):
        target_profile = raw_all[raw_all['Player'] == target_name]
        
        if target_profile.empty:
            st.error("Player not found in database.")
            return
        
        # Prepare search pool
        pool = filtered_df.copy()
        cols_needed = ['Player', 'Squad', 'Position_Group'] + CONFIG['dna_match_features']
        
        # Ensure all columns exist
        for col in cols_needed:
            if col not in pool.columns:
                pool[col] = 0
        
        search_pool = pool[cols_needed].fillna(0).copy()
        
        # Add target if not in pool
        if target_name not in search_pool['Player'].values:
            target_row = target_profile[cols_needed].fillna(0).copy()
            search_pool = pd.concat([search_pool, target_row], ignore_index=True)
        
        # Run similarity search
        results = find_similar_players(target_name, search_pool, CONFIG['dna_match_features'])
        
        if results is None:
            st.error("Could not find target in search pool.")
            return
        
        # Remove self-match
        results = results[results['Player'] != target_name]
        
        # Position filter
        target_pos = target_profile.iloc[0]['Position_Group']
        if match_pos:
            results = results[results['Position_Group'] == target_pos]
        
        if results.empty:
            st.warning(f"No matches found for {target_name}. Try unchecking 'Strict Position Matching'.")
            return
        
        st.success(f"Found {len(results)} matches for {target_name} ({target_pos})")
        
        # Display top 3
        top_matches = results.head(3)
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        
        for i, (idx, row) in enumerate(top_matches.iterrows()):
            original_row = filtered_df[filtered_df['Player'] == row['Player']]
            if not original_row.empty:
                with cols[i]:
                    st.caption(f"Match Quality: {row['Similarity']*100:.1f}%")
                    st.markdown(render_player_card(original_row.iloc[0], prob_col, intuition_score), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
