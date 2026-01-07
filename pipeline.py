"""
Football Superstar Prediction Pipeline
Generates predictions and market valuations for the dashboard

Author: Dale Conaghan
Version: 2.1 (With Empirical Weights)
"""

import pandas as pd
import numpy as np
import os

# === CONFIG ===
CONFIG = {
    'min_minutes': 270,
    'min_minutes_fallback': 90,
    'max_age_prospects': 24,
    'use_empirical_weights': True,  # Set to False to use original heuristic weights
    'data_paths': {
        'raw': 'data/raw/players_data-2025_2026.csv',
        'output': 'data/processed/final_predictions.csv'
    },
    'real_value_anchors': {
        'Lamine Yamal': 168.0,
        'Jude Bellingham': 134.4,
        'Pedri': 117.6,
        'Aleksandar Pavlovic': 54.6,
        'Warren Zaïre-Emery': 42.0,
        'Kobbie Mainoo': 33.6,
        'Gavi': 33.6
    },
    'big_clubs': [
        'paris s-g', 'real madrid', 'barcelona', 'man city', 'liverpool',
        'arsenal', 'chelsea', 'bayern', 'leverkusen', 'inter', 'juventus', 'milan'
    ]
}

# =============================================================================
# EMPIRICALLY DERIVED WEIGHTS
# These weights were learned from historical breakout data using logistic regression.
# Run train_weights.py to regenerate with your own data.
#
# Interpretation:
#   - Positive weights: Higher values predict breakout
#   - Negative weights: Lower values predict breakout (e.g., Age: younger is better)
#   - Magnitude: Larger absolute values = more predictive
# =============================================================================

EMPIRICAL_WEIGHTS = {
    'forward': {
        'SCA90': -0.4698,
        'GCA90': -0.7021,
        'xG_per_90': 1.7294,
        'xAG_per_90': 1.8956,
        'PrgC_per_90': 2.2129,
        'Age': -0.5615,
    },
    'midfielder': {
        'SCA90': -0.2252,
        'GCA90': -0.2133,
        'PrgP_per_90': 1.1939,
        'PrgC_per_90': 0.8151,
        'KP_per_90': 0.6055,
        'Tkl_per_90': -0.0473,
        'Age': -0.3664,
    },
    'defender': {
        'Tkl_per_90': 1.5202,
        'Int_per_90': -1.3071,
        'PrgP_per_90': 2.0047,
        'PrgC_per_90': 0.1089,
        'Blocks_per_90': -1.3570,
        'Age': -0.5166,
    },
    'goalkeeper': {
        'Save_pct': 2.0000,
        'Age': -0.5000,
    },
}

# Original heuristic weights (for comparison)
HEURISTIC_WEIGHTS = {
    'forward': {
        'SCA90': 2.5, 'GCA90': 1.5, 'PrgC_per_90': 1.0, 'xG_per_90': 2.0, 'age_bonus_mult': 0.1
    },
    'midfielder': {
        'PrgP_per_90': 2.0, 'PrgC_per_90': 1.5, 'SCA90': 1.5, 'Tkl_Int_per_90': 0.8, 'age_bonus_mult': 0.1
    },
    'defender': {
        'Tkl_Int_per_90': 1.5, 'Won_pct': 0.1, 'PrgC_per_90': 1.5, 'PrgP_per_90': 1.5, 
        'SCA90': 0.5, 'age_bonus_mult': 0.1
    },
    'goalkeeper': {
        'Save_pct': 0.2, 'age_bonus_mult': 0.1
    },
}


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


def safe_divide(numerator, denominator, default=0):
    """Safe division handling zeros and NaNs."""
    if pd.isna(denominator) or denominator == 0:
        return default
    result = numerator / denominator
    return result if pd.notna(result) else default


def calc_dc_score(row, use_empirical=True):
    """
    Calculate position-aware DC Score for superstar potential.
    
    Args:
        row: DataFrame row with player stats
        use_empirical: If True, use learned weights. If False, use original heuristics.
    
    Returns:
        float: DC Score (higher = more potential)
    """
    pos = row['Position_Group'].lower()
    
    if use_empirical and pos in EMPIRICAL_WEIGHTS:
        return calc_dc_score_empirical(row, pos)
    else:
        return calc_dc_score_heuristic(row, pos)


def calc_dc_score_empirical(row, pos):
    """
    Calculate DC Score using empirically derived weights.
    
    These weights were learned from historical data by analyzing which
    U23 metrics actually predicted future breakout performance.
    """
    weights = EMPIRICAL_WEIGHTS.get(pos, EMPIRICAL_WEIGHTS.get('midfielder', {}))
    
    score = 0
    for feature, weight in weights.items():
        # Get the value, handling missing columns gracefully
        val = row.get(feature, 0)
        if pd.isna(val):
            val = 0
        
        # Age uses raw value (negative weight makes younger = better)
        # Other features are already per-90 normalized
        score += float(val) * weight
    
    return score


def calc_dc_score_heuristic(row, pos):
    """
    Calculate DC Score using original heuristic weights.
    
    This is the original method with hand-tuned weights.
    Kept for comparison and fallback.
    """
    # Age bonus (younger players weighted higher)
    age = row.get('Age', 25)
    age_bonus = (100 - (age * 2)) * 0.1
    
    # Get rate stats with safe defaults
    nineties = row.get('90s', 1)
    if pd.isna(nineties) or nineties == 0:
        nineties = 1
    
    prg_c = row.get('PrgC_per_90', 0) or 0
    prg_p = safe_divide(row.get('PrgP', 0), nineties)
    sca = row.get('SCA90', 0) or 0
    tkl = row.get('Tkl_per_90', 0) or 0
    int_val = row.get('Int_per_90', 0) or 0
    def_actions = tkl + int_val
    won_pct = row.get('Won%', 50) or 50
    save_pct = row.get('Save%', 70) or 70
    gca = row.get('GCA90', 0) or 0
    xg = row.get('xG_per_90', 0) or 0
    
    if pos == 'defender':
        score = (
            (def_actions * 1.5) +
            (won_pct * 0.1) +
            (prg_c * 1.5) +
            (prg_p * 1.5) +
            (sca * 0.5) +
            age_bonus
        )
    elif pos == 'goalkeeper':
        score = (save_pct * 0.2) + age_bonus
    elif pos == 'midfielder':
        score = (
            (prg_p * 2.0) +
            (prg_c * 1.5) +
            (sca * 1.5) +
            (def_actions * 0.8) +
            age_bonus
        )
    else:  # Forward / Unknown
        score = (
            (sca * 2.5) +
            (gca * 1.5) +
            (prg_c * 1.0) +
            (xg * 2.0) +
            age_bonus
        )
    
    return score if pd.notna(score) else 0


def estimate_market_value(row, config):
    """
    Estimate market value based on age, league, club, and DC Index.
    
    Uses real-world anchors for known players and a formula for others.
    """
    player_name = row.get('Player', '')
    
    # Check for real-world anchor values
    if player_name in config['real_value_anchors']:
        return config['real_value_anchors'][player_name]
    
    dc_index = row.get('DC_Index', 50)
    age = row.get('Age', 22)
    comp = str(row.get('Comp', '')).lower()
    squad = str(row.get('Squad', '')).lower()
    position = str(row.get('Position_Group', 'Midfielder'))
    minutes = row.get('Min', 0) or row.get('90s', 0) * 90
    
    # Base value calculation (damped curve)
    base = ((dc_index / 75) ** 3.5) * 18.0
    
    # === 1. AGE MULTIPLIER (wonderkid markup) ===
    if age < 19:
        age_mult = 1.5      # Extreme premium for teenagers
    elif age < 20:
        age_mult = 1.4
    elif age < 22:
        age_mult = 1.2
    elif age > 25:
        age_mult = 0.8
    else:
        age_mult = 1.0
    
    # === 2. LEAGUE MULTIPLIER ===
    if 'premier league' in comp:
        league_mult = 1.5
    elif any(l in comp for l in ['bundesliga', 'la liga', 'serie a']):
        league_mult = 1.2
    elif 'ligue 1' in comp:
        league_mult = 1.1
    else:
        league_mult = 1.0
    
    # === 3. BIG CLUB MULTIPLIER ===
    is_big_club = any(c in squad for c in config['big_clubs'])
    club_mult = 1.3 if is_big_club else 1.0
    
    # === 4. POSITION PREMIUM (NEW) ===
    # Forwards and attacking midfielders command higher fees
    position_upper = position.upper()
    if 'FORWARD' in position_upper or 'FW' in position_upper:
        position_mult = 1.25  # 25% premium for forwards
    elif 'MID' in position_upper:
        position_mult = 1.1   # 10% premium for midfielders
    elif 'DEF' in position_upper:
        position_mult = 0.9   # 10% discount for defenders
    elif 'GOAL' in position_upper or 'GK' in position_upper:
        position_mult = 0.7   # 30% discount for goalkeepers
    else:
        position_mult = 1.0
    
    # === 5. MINUTES CONFIDENCE WEIGHT (NEW) ===
    # More minutes = more proven = more value
    # Discount players with limited game time
    try:
        minutes = float(minutes)
    except (ValueError, TypeError):
        minutes = 500
    
    if minutes >= 2500:
        minutes_mult = 1.1    # Proven starter bonus
    elif minutes >= 1500:
        minutes_mult = 1.0    # Regular player
    elif minutes >= 900:
        minutes_mult = 0.9    # Rotation player discount
    elif minutes >= 500:
        minutes_mult = 0.75   # Limited sample discount
    else:
        minutes_mult = 0.6    # High uncertainty discount
    
    # === 6. TRAJECTORY / FORM (NEW) ===
    # Use Finishing_Alpha as a proxy for form (outperforming xG = hot streak)
    finishing_alpha = row.get('Finishing_Alpha', 0) or 0
    progression_score = row.get('Progression_Score', 0) or 0
    
    # Normalize form indicators
    if finishing_alpha > 3:
        form_mult = 1.15      # Significantly outperforming xG
    elif finishing_alpha > 1:
        form_mult = 1.08      # Slightly hot
    elif finishing_alpha < -3:
        form_mult = 0.9       # Significantly underperforming
    else:
        form_mult = 1.0
    
    # High progression score indicates "engine" player - slight premium
    if progression_score > 15:
        form_mult *= 1.05
    
    # === CALCULATE FINAL VALUE ===
    val = base * age_mult * league_mult * club_mult * position_mult * minutes_mult * form_mult
    
    # Add deterministic "market volatility" based on birth year
    try:
        seed = int(row.get('Born', 2000))
    except (ValueError, TypeError):
        seed = hash(player_name) % 10000
    
    rng = np.random.default_rng(seed)
    randomness = rng.uniform(0.95, 1.05)  # Reduced randomness (was 0.9-1.1)
    final_val = val * randomness
    
    # Apply floor values
    floor = 15.0 if is_big_club else 1.0
    
    return round(max(floor, final_val), 1)


def run_pipeline():
    """Main pipeline execution."""
    print("🚀 Starting 2025-2026 Prediction Pipeline...")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, CONFIG['data_paths']['raw'])
    output_path = os.path.join(base_dir, CONFIG['data_paths']['output'])
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Load Data
    print(f"📂 Loading data: {raw_path}")
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {raw_path}")
        print("Please ensure the raw data file exists.")
        return None
    
    # 2. Basic Cleaning
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(25)
    df['Position_Group'] = df['Pos'].apply(get_pos_group)
    
    # Clean Won% if it's a string percentage
    if 'Won%' in df.columns:
        df['Won%'] = pd.to_numeric(
            df['Won%'].astype(str).str.replace('%', ''),
            errors='coerce'
        ).fillna(50)
    
    # 3. Normalize stats per 90
    if '90s' in df.columns:
        df['90s'] = pd.to_numeric(df['90s'], errors='coerce').fillna(0)
        
        cols_to_normalize = ['Gls', 'Ast', 'xG', 'xAG', 'PrgC', 'PrgP', 'PrgR', 'KP', 'Tkl', 'Int', 'PPA', 'Carries']
        for col in cols_to_normalize:
            if col in df.columns:
                df[f'{col}_per_90'] = df[col] / df['90s'].replace(0, 1)
    
    # 4. Filter for minimum minutes
    print(f"⏳ Filtering for reliable data (Min {CONFIG['min_minutes']} mins)...")
    df_filtered = df[df['Min'] >= CONFIG['min_minutes']].copy()
    
    if df_filtered.empty:
        print(f"⚠️ No players with >{CONFIG['min_minutes']} mins. Using {CONFIG['min_minutes_fallback']} mins.")
        df_filtered = df[df['Min'] >= CONFIG['min_minutes_fallback']].copy()
    
    if df_filtered.empty:
        print("❌ Error: No players meet minimum minutes threshold.")
        return None
    
    # 5. Calculate DC Score
    print("🧠 Calculating DC Scores...")
    use_empirical = CONFIG.get('use_empirical_weights', True)
    method = "empirical" if use_empirical else "heuristic"
    print(f"   Using {method} weights")
    
    df_filtered['DC_Score'] = df_filtered.apply(
        lambda row: calc_dc_score(row, use_empirical=use_empirical), 
        axis=1
    )
    
    # 6. Calculate Proprietary Metrics
    # Progression Score (per 90)
    prgc = df_filtered['PrgC'] if 'PrgC' in df_filtered.columns else pd.Series(0, index=df_filtered.index)
    prgp = df_filtered['PrgP'] if 'PrgP' in df_filtered.columns else pd.Series(0, index=df_filtered.index)
    df_filtered['Progression_Score'] = (prgc.fillna(0) + prgp.fillna(0)) / df_filtered['90s'].replace(0, 1)
    
    # Finishing Alpha (Goals - xG)
    gls = df_filtered['Gls'] if 'Gls' in df_filtered.columns else pd.Series(0, index=df_filtered.index)
    xg = df_filtered['xG'] if 'xG' in df_filtered.columns else pd.Series(0, index=df_filtered.index)
    df_filtered['Finishing_Alpha'] = gls.fillna(0) - xg.fillna(0)
    
    # Ghost Factor (Threat per Touch)
    sca90 = df_filtered['SCA90'] if 'SCA90' in df_filtered.columns else pd.Series(0, index=df_filtered.index)
    touches = df_filtered['Touches'] if 'Touches' in df_filtered.columns else pd.Series(1, index=df_filtered.index)
    df_filtered['Ghost_Factor'] = np.where(
        touches > 0,
        (sca90.fillna(0) / touches.replace(0, 1)) * 100,
        0
    )
    
    # 7. Normalize DC_Index to Percentile Rank (0-100) WITHIN POSITION GROUP
    # This ensures goalkeepers are compared to goalkeepers, not outfield players
    df_filtered['DC_Index'] = df_filtered.groupby('Position_Group')['DC_Score'].transform(
        lambda x: x.rank(pct=True) * 100
    )
    
    # 8. Filter for U24 Prospects
    prospects = df_filtered[df_filtered['Age'] <= CONFIG['max_age_prospects']].copy()
    print(f"📊 Found {len(prospects)} U{CONFIG['max_age_prospects']} prospects")
    
    # 9. Estimate Market Values
    print("💰 Projecting market values...")
    prospects['Est_Value'] = prospects.apply(lambda row: estimate_market_value(row, CONFIG), axis=1)
    
    # 10. Calculate Value Score (ROI metric)
    prospects['Value_Score'] = prospects['DC_Index'] * (100 / (100 + prospects['Est_Value']))
    
    # 11. Rename columns for app compatibility
    rename_map = {
        'Gls_per_90': 'Goals_per_90',
        'Ast_per_90': 'Assists_per_90'
    }
    prospects = prospects.rename(columns=rename_map)
    
    # 12. Final NaN cleanup
    numeric_cols = ['DC_Index', 'DC_Score', 'Progression_Score', 'Finishing_Alpha', 
                    'Ghost_Factor', 'Est_Value', 'Value_Score', 'Goals_per_90', 'Assists_per_90']
    for col in numeric_cols:
        if col in prospects.columns:
            prospects[col] = prospects[col].fillna(0)
    
    # 13. Save
    print(f"💾 Saving predictions to: {output_path}")
    prospects.to_csv(output_path, index=False)
    
    print(f"✅ Pipeline complete. {len(prospects)} prospects identified.")
    return prospects


if __name__ == "__main__":
    run_pipeline()
