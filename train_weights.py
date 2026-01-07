"""
Empirical Weight Training for DC Index
======================================

This script:
1. Loads historical player data (multiple seasons)
2. Identifies players who "broke out" (became elite)
3. Trains a model to find which U23 metrics predict breakouts
4. Outputs empirically-derived weights for the DC Index

Author: Dale Conaghan
For: NCI Higher Diploma in Data Analytics
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
import os
import time

warnings.filterwarnings('ignore')

# === CONFIG ===
CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'min_minutes': 500,  # Minimum minutes to be considered
    'max_age_training': 23,  # U23 players only for training
    'breakout_definitions': {
        'transfer_fee_threshold': 25,  # £M - got a significant transfer
        'peak_ga_threshold': 12,  # G+A in a single season
        'intl_caps_threshold': 5,  # Senior international caps
        'top_league_minutes_threshold': 2000,  # Minutes in top 5 league by age 26
    },
    'features': {
        'forward': [
            'SCA90', 'GCA90', 'xG_per_90', 'xAG_per_90', 'PrgC_per_90', 
            'PrgR_per_90', 'Touches_per_90', 'Carries_per_90', 'Age'
        ],
        'midfielder': [
            'SCA90', 'GCA90', 'PrgP_per_90', 'PrgC_per_90', 'KP_per_90',
            'Tkl_per_90', 'Int_per_90', 'Cmp_pct', 'Age'
        ],
        'defender': [
            'Tkl_per_90', 'Int_per_90', 'Blocks_per_90', 'Clr_per_90',
            'PrgP_per_90', 'PrgC_per_90', 'Won_pct', 'Age'
        ],
        'goalkeeper': [
            'Save_pct', 'CS_pct', 'PSxG_diff', 'Launch_pct', 
            'AvgLen', 'Stp_pct', 'Age'
        ]
    }
}


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_fbref_data(filepath):
    """
    Load FBRef data from CSV.
    
    Expected columns vary by source, but we'll standardize them.
    Handles various encodings for player names with accents.
    """
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  ⚠️ Error with {encoding}: {e}")
            continue
    
    if df is None:
        raise ValueError(f"Could not read {filepath} with any encoding")
    
    # FORCE unique columns - rename duplicates with suffix
    seen = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    
    # Standardize column names (FBRef uses various formats)
    column_mapping = {
        # Standard stats
        'Gls': 'Goals', 'Ast': 'Assists', 'G+A': 'GA',
        'G-PK': 'Goals_noPK', 'xG': 'xG', 'xAG': 'xAG',
        
        # Per 90 stats (some datasets have these pre-calculated)
        'Gls.1': 'Goals_per_90', 'Ast.1': 'Assists_per_90',
        
        # Possession
        'Touches': 'Touches', 'Carries': 'Carries',
        'PrgC': 'PrgC', 'PrgP': 'PrgP', 'PrgR': 'PrgR',
        
        # Passing
        'Cmp%': 'Cmp_pct', 'KP': 'KP',
        
        # Defense
        'Tkl': 'Tkl', 'Int': 'Int', 'Blocks': 'Blocks', 'Clr': 'Clr',
        
        # Aerials
        'Won%': 'Won_pct',
        
        # GK specific
        'Save%': 'Save_pct', 'CS%': 'CS_pct', 'PSxG+/-': 'PSxG_diff',
        'Launch%': 'Launch_pct', 'AvgLen': 'AvgLen', 'Stp%': 'Stp_pct',
        
        # Creation
        'SCA90': 'SCA90', 'GCA90': 'GCA90',
        'SCA': 'SCA', 'GCA': 'GCA'
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    return df


def calculate_per90_stats(df):
    """Calculate per-90 stats from totals."""
    df = df.copy()
    
    # Ensure 90s column exists
    if '90s' not in df.columns and 'Min' in df.columns:
        df['90s'] = df['Min'] / 90
    
    nineties = df['90s'].replace(0, np.nan)
    
    # Stats to normalize
    total_stats = ['Goals', 'Assists', 'xG', 'xAG', 'PrgC', 'PrgP', 'PrgR', 
                   'Touches', 'Carries', 'KP', 'Tkl', 'Int', 'Blocks', 'Clr',
                   'SCA', 'GCA']
    
    for stat in total_stats:
        if stat in df.columns:
            per90_col = f'{stat}_per_90'
            if per90_col not in df.columns:
                df[per90_col] = df[stat] / nineties
    
    # SCA90 and GCA90 might already exist
    if 'SCA90' not in df.columns and 'SCA_per_90' in df.columns:
        df['SCA90'] = df['SCA_per_90']
    if 'GCA90' not in df.columns and 'GCA_per_90' in df.columns:
        df['GCA90'] = df['GCA_per_90']
    
    return df


def get_position_group(pos):
    """Map position codes to groups."""
    if pd.isna(pos):
        return 'Unknown'
    pos = str(pos).upper()
    if 'GK' in pos:
        return 'goalkeeper'
    if 'DF' in pos or 'CB' in pos or 'FB' in pos or 'WB' in pos:
        return 'defender'
    if 'MF' in pos or 'CM' in pos or 'DM' in pos or 'AM' in pos:
        return 'midfielder'
    if 'FW' in pos or 'ST' in pos or 'CF' in pos or 'LW' in pos or 'RW' in pos:
        return 'forward'
    return 'midfielder'  # Default assumption


def prepare_training_data(df_historical, df_outcomes=None):
    """
    Prepare data for model training.
    
    Args:
        df_historical: Historical U23 player stats
        df_outcomes: Optional separate dataframe with career outcomes
    """
    df = df_historical.copy()
    
    # Calculate per-90 stats
    df = calculate_per90_stats(df)
    
    # Add position groups
    if 'Position_Group' not in df.columns:
        df['Position_Group'] = df['Pos'].apply(get_position_group)
    
    # Filter for minimum minutes and age
    df = df[df['Min'] >= CONFIG['min_minutes']].copy()
    df = df[df['Age'] <= CONFIG['max_age_training']].copy()
    
    return df


# =============================================================================
# BREAKOUT LABELING
# =============================================================================

def create_breakout_labels_from_same_dataset(df_early, df_peak):
    """
    Create breakout labels by comparing early career stats to peak stats.
    
    This is for when you have the same players across multiple seasons.
    
    Args:
        df_early: DataFrame with U23 stats (training features)
        df_peak: DataFrame with later career stats (for labeling)
    """
    # Get peak stats per player
    peak_stats = df_peak.groupby('Player').agg({
        'Goals': 'max',
        'Assists': 'max',
        'Min': 'sum'
    }).reset_index()
    
    peak_stats.columns = ['Player', 'Peak_Goals', 'Peak_Assists', 'Total_Min']
    peak_stats['Peak_GA'] = peak_stats['Peak_Goals'] + peak_stats['Peak_Assists']
    
    # Merge with early data
    df = df_early.merge(peak_stats, on='Player', how='left')
    
    # Define breakout
    threshold = CONFIG['breakout_definitions']['peak_ga_threshold']
    df['Broke_Out'] = (df['Peak_GA'] >= threshold).astype(int)
    
    return df


def create_breakout_labels_manual(df, known_breakouts):
    """
    Create labels from a manual list of players who broke out.
    
    Args:
        df: DataFrame with player stats
        known_breakouts: List of player names who became elite
    """
    df = df.copy()
    df['Broke_Out'] = df['Player'].isin(known_breakouts).astype(int)
    return df


def create_synthetic_labels(df):
    """
    Create synthetic breakout labels based on exceptional U23 performance.
    
    This is a bootstrap method when you don't have outcome data.
    Uses the assumption that U23s already performing at elite levels
    are likely to break out (which is circular but useful for testing).
    """
    df = df.copy()
    
    # Calculate composite score
    df['temp_score'] = 0
    
    for pos in df['Position_Group'].unique():
        mask = df['Position_Group'] == pos
        pos_df = df[mask].copy()
        
        if pos == 'forward':
            score_cols = ['SCA90', 'GCA90', 'xG_per_90']
        elif pos == 'midfielder':
            score_cols = ['SCA90', 'PrgP_per_90', 'PrgC_per_90']
        elif pos == 'defender':
            score_cols = ['Tkl_per_90', 'Int_per_90', 'PrgP_per_90']
        else:
            score_cols = ['Save_pct'] if 'Save_pct' in df.columns else []
        
        available_cols = [c for c in score_cols if c in df.columns]
        
        if available_cols:
            for col in available_cols:
                # Percentile rank within position
                df.loc[mask, 'temp_score'] += df.loc[mask, col].rank(pct=True)
            df.loc[mask, 'temp_score'] /= len(available_cols)
    
    # Top 15% are "breakouts"
    df['Broke_Out'] = (df['temp_score'] >= df['temp_score'].quantile(0.85)).astype(int)
    df = df.drop(columns=['temp_score'])
    
    return df


# =============================================================================
# KNOWN BREAKOUT PLAYERS (2018-2023 verification set)
# =============================================================================

KNOWN_BREAKOUTS = {
    # Players who were U23 in 2019-2021 and became elite by 2024
    'forwards': [
        'Erling Haaland', 'Kylian Mbappé', 'Vinícius Júnior', 'Bukayo Saka',
        'Phil Foden', 'Jadon Sancho', 'Ansu Fati', 'Pedri', 'Jude Bellingham',
        'Florian Wirtz', 'Jamal Musiala', 'Lamine Yamal', 'Rodrygo',
        'Rafael Leão', 'Khvicha Kvaratskhelia', 'Mohammed Kudus',
        'Michael Olise', 'Cole Palmer', 'Alejandro Garnacho',
        'Harvey Elliott', 'Gabriel Martinelli', 'Darwin Núñez'
    ],
    'midfielders': [
        'Pedri', 'Jude Bellingham', 'Florian Wirtz', 'Jamal Musiala',
        'Eduardo Camavinga', 'Ryan Gravenberch', 'Aurélien Tchouaméni',
        'Gavi', 'Warren Zaïre-Emery', 'Kobbie Mainoo', 'Enzo Fernández',
        'Romeo Lavia', 'Moisés Caicedo', 'Declan Rice', 'Vitinha',
        'Sandro Tonali', 'Nicolò Barella'
    ],
    'defenders': [
        'Alphonso Davies', 'Theo Hernández', 'Achraf Hakimi', 'Josko Gvardiol',
        'Jurriën Timber', 'William Saliba', 'Lisandro Martínez',
        'Castello Lukeba', 'Leny Yoro', 'Gonçalo Inácio', 'Destiny Udogie',
        'Malo Gusto', 'Jeremie Frimpong', 'Rico Lewis'
    ],
    'goalkeepers': [
        'Gianluigi Donnarumma', 'Maignan', 'Diogo Costa', 'Giorgi Mamardashvili',
        'Gavin Bazunu', 'Lucas Chevalier'
    ]
}

# Flatten for easy lookup
ALL_KNOWN_BREAKOUTS = (
    KNOWN_BREAKOUTS['forwards'] + 
    KNOWN_BREAKOUTS['midfielders'] + 
    KNOWN_BREAKOUTS['defenders'] + 
    KNOWN_BREAKOUTS['goalkeepers']
)


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_position_model(df, position, model_type='logistic'):
    """
    Train a model for a specific position group.
    
    Returns:
        dict with model, scaler, weights, and metrics
    """
    pos_df = df[df['Position_Group'] == position].copy()
    
    if len(pos_df) < 50:
        print(f"  ⚠️  Not enough {position} data ({len(pos_df)} rows). Skipping.")
        return None
    
    # Get features for this position
    features = CONFIG['features'].get(position, CONFIG['features']['midfielder'])
    available_features = [f for f in features if f in pos_df.columns]
    
    if len(available_features) < 3:
        print(f"  ⚠️  Not enough features available for {position}. Skipping.")
        return None
    
    # Prepare X and y
    X = pos_df[available_features].fillna(0)
    y = pos_df['Broke_Out']
    
    # Check class balance
    breakout_rate = y.mean()
    print(f"  📊 {position}: {len(pos_df)} players, {y.sum()} breakouts ({breakout_rate:.1%})")
    
    if y.sum() < 10:
        print(f"  ⚠️  Too few breakout examples for {position}. Using synthetic boost.")
        # Oversample breakouts for training stability
        breakout_indices = pos_df[pos_df['Broke_Out'] == 1].index
        X = pd.concat([X, X.loc[breakout_indices].sample(n=min(20, len(breakout_indices)*3), replace=True)])
        y = pd.concat([y, y.loc[breakout_indices].sample(n=min(20, len(breakout_indices)*3), replace=True)])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'],
        stratify=y if y.sum() >= 2 else None
    )
    
    # Train model
    if model_type == 'logistic':
        model = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            random_state=CONFIG['random_state']
        )
    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight='balanced',
            random_state=CONFIG['random_state']
        )
    elif model_type == 'gb':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=CONFIG['random_state']
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=min(CONFIG['cv_folds'], y.sum()), scoring='roc_auc')
    
    # Extract weights/importance
    if model_type == 'logistic':
        weights = dict(zip(available_features, model.coef_[0]))
    else:
        weights = dict(zip(available_features, model.feature_importances_))
    
    # Calculate metrics
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
    
    results = {
        'model': model,
        'scaler': scaler,
        'features': available_features,
        'weights': weights,
        'metrics': {
            'auc': auc,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'n_samples': len(pos_df),
            'n_breakouts': int(y.sum()),
            'breakout_rate': breakout_rate
        }
    }
    
    return results


def train_all_models(df, model_type='logistic'):
    """Train models for all positions."""
    print(f"\n{'='*60}")
    print(f"TRAINING {model_type.upper()} MODELS")
    print(f"{'='*60}\n")
    
    results = {}
    
    for position in ['forward', 'midfielder', 'defender', 'goalkeeper']:
        print(f"\n🎯 Training {position} model...")
        result = train_position_model(df, position, model_type)
        if result:
            results[position] = result
    
    return results


def print_results(results):
    """Pretty print the training results."""
    print(f"\n{'='*60}")
    print("EMPIRICAL WEIGHTS RESULTS")
    print(f"{'='*60}\n")
    
    for position, data in results.items():
        print(f"\n{'─'*40}")
        print(f"📊 {position.upper()}")
        print(f"{'─'*40}")
        
        metrics = data['metrics']
        print(f"   Samples: {metrics['n_samples']} | Breakouts: {metrics['n_breakouts']} ({metrics['breakout_rate']:.1%})")
        print(f"   AUC: {metrics['auc']:.3f} | CV AUC: {metrics['cv_auc_mean']:.3f} (±{metrics['cv_auc_std']:.3f})")
        
        print(f"\n   Feature Weights (sorted by importance):")
        sorted_weights = sorted(data['weights'].items(), key=lambda x: -abs(x[1]))
        for feature, weight in sorted_weights:
            direction = "↑" if weight > 0 else "↓"
            print(f"      {direction} {feature}: {weight:+.4f}")


def generate_pipeline_code(results):
    """Generate Python code for the updated pipeline with empirical weights."""
    
    code = '''
# =============================================================================
# EMPIRICALLY DERIVED WEIGHTS
# Generated by train_weights.py
# =============================================================================

EMPIRICAL_WEIGHTS = {
'''
    
    for position, data in results.items():
        code += f"    '{position}': {{\n"
        for feature, weight in data['weights'].items():
            code += f"        '{feature}': {weight:.4f},\n"
        code += f"    }},\n"
    
    code += '''}\n

def calc_dc_score_empirical(row, weights_dict):
    """
    Calculate DC Score using empirically derived weights.
    
    This replaces the arbitrary weights with coefficients learned
    from historical breakout data.
    """
    pos = row['Position_Group'].lower()
    weights = weights_dict.get(pos, weights_dict.get('midfielder', {}))
    
    score = 0
    for feature, weight in weights.items():
        if feature == 'Age':
            # Age is typically negative (younger = better)
            val = row.get(feature, 22)
        else:
            val = row.get(feature, 0) or 0
        
        score += val * weight
    
    return score
'''
    
    return code


# =============================================================================
# DATA SCRAPING (Optional - for fetching fresh data)
# =============================================================================

def scrape_fbref_season(season='2023-2024', league='Big5'):
    """
    Scrape FBRef data for a season.
    
    NOTE: This requires requests and BeautifulSoup.
    FBRef may rate-limit or block scrapers - be respectful.
    
    For the assignment, it's easier to download CSVs manually from:
    https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("❌ requests and beautifulsoup4 required for scraping")
        print("   pip install requests beautifulsoup4")
        return None
    
    print(f"⏳ Scraping FBRef data for {season}...")
    
    # FBRef Big 5 leagues stats page
    url = f"https://fbref.com/en/comps/Big5/{season}/stats/players/{season}-Big-5-European-Leagues-Stats"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse tables
        tables = pd.read_html(response.text)
        
        # The main stats table is usually the largest one
        df = max(tables, key=len)
        
        # Clean up multi-level headers if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns]
        
        print(f"✅ Scraped {len(df)} player records")
        return df
        
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        print("   Try downloading CSVs manually from FBRef")
        return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def create_sample_data():
    """
    Create sample training data if real data isn't available.
    
    This generates synthetic data that mimics the structure of real FBRef data.
    USE REAL DATA FOR YOUR ACTUAL SUBMISSION.
    """
    print("⚠️  Creating synthetic sample data for demonstration...")
    print("   For your NCI submission, use real FBRef data!\n")
    
    np.random.seed(42)
    n_players = 500
    
    # Generate random player data
    data = {
        'Player': [f'Player_{i}' for i in range(n_players)],
        'Age': np.random.randint(17, 24, n_players),
        'Pos': np.random.choice(['FW', 'MF', 'DF', 'GK'], n_players, p=[0.25, 0.35, 0.30, 0.10]),
        'Min': np.random.randint(500, 3000, n_players),
        '90s': None,  # Will calculate
        
        # Attacking
        'Goals': np.random.poisson(5, n_players),
        'Assists': np.random.poisson(3, n_players),
        'xG': np.random.uniform(0, 15, n_players),
        'xAG': np.random.uniform(0, 8, n_players),
        'SCA': np.random.poisson(40, n_players),
        'GCA': np.random.poisson(8, n_players),
        
        # Progression
        'PrgC': np.random.poisson(50, n_players),
        'PrgP': np.random.poisson(80, n_players),
        'PrgR': np.random.poisson(60, n_players),
        
        # Possession
        'Touches': np.random.poisson(1000, n_players),
        'Carries': np.random.poisson(400, n_players),
        'KP': np.random.poisson(30, n_players),
        
        # Defense
        'Tkl': np.random.poisson(30, n_players),
        'Int': np.random.poisson(20, n_players),
        'Blocks': np.random.poisson(15, n_players),
        'Clr': np.random.poisson(40, n_players),
        
        # Passing
        'Cmp%': np.random.uniform(65, 92, n_players),
        
        # Aerials
        'Won%': np.random.uniform(30, 70, n_players),
        
        # GK
        'Save%': np.random.uniform(60, 80, n_players),
        'CS%': np.random.uniform(20, 40, n_players),
    }
    
    df = pd.DataFrame(data)
    df['90s'] = df['Min'] / 90
    
    # Add some known "breakout" players
    breakout_names = [
        'Jude Bellingham', 'Pedri', 'Jamal Musiala', 'Florian Wirtz',
        'Bukayo Saka', 'Phil Foden', 'Vinícius Júnior', 'Alphonso Davies',
        'Eduardo Camavinga', 'Gavi', 'William Saliba', 'Josko Gvardiol'
    ]
    
    for i, name in enumerate(breakout_names):
        if i < len(df):
            df.loc[i, 'Player'] = name
            # Boost their stats to make them stand out
            df.loc[i, 'Goals'] *= 2
            df.loc[i, 'Assists'] *= 2
            df.loc[i, 'SCA'] *= 1.5
            df.loc[i, 'PrgC'] *= 1.5
    
    return df


def main():
    """Main execution function."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     DC INDEX - EMPIRICAL WEIGHT TRAINING                     ║
║     Finding what actually predicts football breakouts        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check for existing data files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    
    historical_files = [
        'players_data-2019_2020.csv',
        'players_data-2020_2021.csv',
        'players_data-2021_2022.csv',
        'players_data-2022_2023.csv',
        'players_data-2023_2024.csv',
        'players_data-2024_2025.csv',
        'players_data-2025_2026.csv',
    ]
    
    # Try to load real data
    all_data = []
    for filename in historical_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"📂 Loading {filename}...")
            df = load_fbref_data(filepath)
            # Extract season from filename
            season = filename.replace('players_data-', '').replace('.csv', '')
            df['Season'] = season
            all_data.append(df)
    
    if all_data:
        # Concat with outer join to handle different column sets
        # Then drop any fully duplicate columns that might remain
        df = pd.concat(all_data, ignore_index=True, sort=False)
        
        # Remove any remaining duplicate columns (keep first)
        df = df.loc[:, ~df.columns.duplicated()]
        
        print(f"\n✅ Loaded {len(df)} total player-seasons from {len(all_data)} files")
    else:
        print("\n⚠️  No historical data files found in data/raw/")
        print("   Using synthetic sample data for demonstration.\n")
        df = create_sample_data()
    
    # Prepare training data
    print("\n⏳ Preparing training data...")
    df = prepare_training_data(df)
    
    # Add position groups
    df['Position_Group'] = df['Pos'].apply(get_position_group)
    
    # Create breakout labels
    print("⏳ Creating breakout labels...")
    
    # Method 1: Use known breakouts list
    df = create_breakout_labels_manual(df, ALL_KNOWN_BREAKOUTS)
    
    # If not enough labeled, supplement with synthetic
    if df['Broke_Out'].sum() < 30:
        print("   Adding synthetic labels to supplement...")
        df_synthetic = create_synthetic_labels(df.copy())
        # Combine: real labels take precedence
        df.loc[df['Broke_Out'] == 0, 'Broke_Out'] = df_synthetic.loc[df['Broke_Out'] == 0, 'Broke_Out']
    
    print(f"   Total breakouts labeled: {df['Broke_Out'].sum()} ({df['Broke_Out'].mean():.1%})")
    
    # Train models
    print("\n" + "="*60)
    
    # Try multiple model types
    all_results = {}
    
    for model_type in ['logistic', 'rf']:
        results = train_all_models(df, model_type=model_type)
        all_results[model_type] = results
        print_results(results)
    
    # Generate code for best model (logistic for interpretability)
    print("\n" + "="*60)
    print("GENERATED CODE FOR PIPELINE")
    print("="*60)
    
    best_results = all_results.get('logistic', all_results.get('rf', {}))
    code = generate_pipeline_code(best_results)
    print(code)
    
    # Save to file
    output_path = os.path.join(base_dir, 'empirical_weights.py')
    with open(output_path, 'w') as f:
        f.write(code)
    print(f"\n💾 Weights saved to: {output_path}")
    
    # Save full results
    results_path = os.path.join(base_dir, 'data', 'processed', 'model_training_results.csv')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    results_data = []
    for model_type, positions in all_results.items():
        for position, data in positions.items():
            for feature, weight in data['weights'].items():
                results_data.append({
                    'model_type': model_type,
                    'position': position,
                    'feature': feature,
                    'weight': weight,
                    'auc': data['metrics']['auc'],
                    'cv_auc': data['metrics']['cv_auc_mean']
                })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_path, index=False)
    print(f"💾 Full results saved to: {results_path}")
    
    print("\n✅ Training complete!")
    print("\nNext steps:")
    print("   1. Review the weights - do they make intuitive sense?")
    print("   2. Copy the EMPIRICAL_WEIGHTS dict to your pipeline.py")
    print("   3. Replace calc_dc_score() with calc_dc_score_empirical()")
    print("   4. Re-run the pipeline and dashboard")
    
    return all_results


if __name__ == "__main__":
    results = main()
