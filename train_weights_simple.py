"""
Empirical Weight Training for DC Index (Simplified Version)
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
import warnings
import os

warnings.filterwarnings('ignore')

# === CONFIG ===
CONFIG = {
    'random_state': 42,
    'min_minutes': 500,
    'max_age_training': 23,
}

# Known breakout players (U23 who became elite)
KNOWN_BREAKOUTS = [
    'Erling Haaland', 'Kylian Mbappé', 'Vinícius Júnior', 'Bukayo Saka',
    'Phil Foden', 'Jadon Sancho', 'Pedri', 'Jude Bellingham',
    'Florian Wirtz', 'Jamal Musiala', 'Lamine Yamal', 'Rodrygo',
    'Rafael Leão', 'Khvicha Kvaratskhelia', 'Cole Palmer',
    'Eduardo Camavinga', 'Gavi', 'Warren Zaïre-Emery', 'Kobbie Mainoo',
    'Alphonso Davies', 'Theo Hernández', 'Josko Gvardiol', 'William Saliba',
    'Aurélien Tchouaméni', 'Moisés Caicedo', 'Enzo Fernández',
    'Gabriel Martinelli', 'Darwin Núñez', 'Michael Olise',
    'Alejandro Garnacho', 'Harvey Elliott', 'Rico Lewis'
]


def load_single_csv(filepath):
    """Load a single CSV with encoding handling and unique columns."""
    # Try encodings
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
            break
        except:
            continue
    else:
        raise ValueError(f"Could not read {filepath}")
    
    # Force unique column names
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
    
    return df


def get_position_group(pos):
    """Map position codes to groups."""
    if pd.isna(pos):
        return 'unknown'
    pos = str(pos).upper()
    if 'GK' in pos:
        return 'goalkeeper'
    if 'DF' in pos or 'CB' in pos or 'FB' in pos or 'WB' in pos:
        return 'defender'
    if 'MF' in pos or 'CM' in pos or 'DM' in pos or 'AM' in pos:
        return 'midfielder'
    if 'FW' in pos or 'ST' in pos or 'CF' in pos or 'LW' in pos or 'RW' in pos:
        return 'forward'
    return 'midfielder'


def process_dataframe(df):
    """Add calculated columns to a dataframe."""
    df = df.copy()
    
    # Find minutes column
    if 'Min' in df.columns:
        df['Min'] = pd.to_numeric(df['Min'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df['90s'] = df['Min'] / 90
    elif '90s' in df.columns:
        df['90s'] = pd.to_numeric(df['90s'], errors='coerce').fillna(1)
    else:
        df['90s'] = 10  # Default
    
    df['90s'] = df['90s'].replace(0, 1)
    
    # Age
    if 'Age' in df.columns:
        # Handle "25-123" format (age-days)
        df['Age'] = df['Age'].astype(str).str.split('-').str[0]
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(25)
    
    # Position
    pos_col = 'Pos' if 'Pos' in df.columns else 'Position' if 'Position' in df.columns else None
    if pos_col:
        df['Position_Group'] = df[pos_col].apply(get_position_group)
    else:
        df['Position_Group'] = 'midfielder'
    
    # Calculate per-90 stats
    stats_to_normalize = {
        'Gls': 'Goals_per_90', 'Goals': 'Goals_per_90',
        'Ast': 'Assists_per_90', 'Assists': 'Assists_per_90',
        'xG': 'xG_per_90', 'xAG': 'xAG_per_90',
        'PrgC': 'PrgC_per_90', 'PrgP': 'PrgP_per_90', 'PrgR': 'PrgR_per_90',
        'Touches': 'Touches_per_90', 'Carries': 'Carries_per_90',
        'KP': 'KP_per_90', 'Tkl': 'Tkl_per_90', 'Int': 'Int_per_90',
        'Blocks': 'Blocks_per_90', 'Clr': 'Clr_per_90',
        'SCA': 'SCA_per_90', 'GCA': 'GCA_per_90'
    }
    
    for raw_col, per90_col in stats_to_normalize.items():
        if raw_col in df.columns and per90_col not in df.columns:
            df[per90_col] = pd.to_numeric(df[raw_col], errors='coerce').fillna(0) / df['90s']
    
    # SCA90 / GCA90 aliases
    if 'SCA90' not in df.columns:
        df['SCA90'] = df.get('SCA_per_90', 0)
    if 'GCA90' not in df.columns:
        df['GCA90'] = df.get('GCA_per_90', 0)
    
    return df


def train_position_model(df, position):
    """Train model for a position."""
    pos_df = df[df['Position_Group'] == position].copy()
    
    if len(pos_df) < 30:
        print(f"  ⚠️ Not enough {position} data ({len(pos_df)} rows)")
        return None
    
    # Define features by position
    if position == 'forward':
        features = ['SCA90', 'GCA90', 'xG_per_90', 'xAG_per_90', 'PrgC_per_90', 'Age']
    elif position == 'midfielder':
        features = ['SCA90', 'GCA90', 'PrgP_per_90', 'PrgC_per_90', 'KP_per_90', 'Tkl_per_90', 'Age']
    elif position == 'defender':
        features = ['Tkl_per_90', 'Int_per_90', 'PrgP_per_90', 'PrgC_per_90', 'Blocks_per_90', 'Age']
    else:
        return None
    
    # Filter to available features
    available = [f for f in features if f in pos_df.columns]
    if len(available) < 3:
        print(f"  ⚠️ Not enough features for {position}")
        return None
    
    # Prepare data
    X = pos_df[available].fillna(0)
    y = pos_df['Broke_Out']
    
    print(f"  📊 {position}: {len(pos_df)} players, {y.sum()} breakouts ({y.mean():.1%})")
    
    if y.sum() < 5:
        print(f"  ⚠️ Too few breakouts for {position}")
        return None
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    
    # Get weights
    weights = dict(zip(available, model.coef_[0]))
    
    # Evaluate
    try:
        cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='roc_auc')
        auc = cv_scores.mean()
    except:
        auc = 0.5
    
    return {
        'weights': weights,
        'auc': auc,
        'n_samples': len(pos_df),
        'n_breakouts': int(y.sum())
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     DC INDEX - EMPIRICAL WEIGHT TRAINING                     ║
║     Finding what actually predicts football breakouts        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    
    # Find CSV files
    csv_files = [
        'players_data-2021_2022.csv',
        'players_data-2022_2023.csv',
        'players_data-2023_2024.csv',
        'players_data-2024_2025.csv',
        'players_data-2025_2026.csv',
    ]
    
    # Load and process each file separately, then combine
    all_dfs = []
    for filename in csv_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"📂 Loading {filename}...")
            try:
                df = load_single_csv(filepath)
                df = process_dataframe(df)
                df['Season'] = filename.replace('players_data-', '').replace('.csv', '')
                all_dfs.append(df)
            except Exception as e:
                print(f"  ⚠️ Error loading {filename}: {e}")
    
    if not all_dfs:
        print("❌ No data files found!")
        return
    
    # Combine using a simple row-by-row approach
    print(f"\n⏳ Combining {len(all_dfs)} files...")
    
    # Get all unique columns
    all_cols = set()
    for df in all_dfs:
        all_cols.update(df.columns)
    all_cols = list(all_cols)
    
    # Reindex each dataframe to have all columns
    combined_rows = []
    for df in all_dfs:
        for col in all_cols:
            if col not in df.columns:
                df[col] = np.nan
        combined_rows.append(df[all_cols])
    
    # Simple concat
    df = pd.concat(combined_rows, axis=0, ignore_index=True)
    
    print(f"✅ Loaded {len(df)} total player-seasons")
    
    # Filter
    print("\n⏳ Preparing training data...")
    if 'Min' in df.columns:
        df = df[pd.to_numeric(df['Min'].astype(str).str.replace(',', ''), errors='coerce').fillna(0) >= CONFIG['min_minutes']]
    if 'Age' in df.columns:
        df = df[pd.to_numeric(df['Age'], errors='coerce').fillna(99) <= CONFIG['max_age_training']]
    
    print(f"   Filtered to {len(df)} U23 players with {CONFIG['min_minutes']}+ minutes")
    
    # Label breakouts
    print("⏳ Creating breakout labels...")
    if 'Player' in df.columns:
        df['Broke_Out'] = df['Player'].isin(KNOWN_BREAKOUTS).astype(int)
    else:
        # Try to find player column
        player_cols = [c for c in df.columns if 'player' in c.lower() or 'name' in c.lower()]
        if player_cols:
            df['Broke_Out'] = df[player_cols[0]].isin(KNOWN_BREAKOUTS).astype(int)
        else:
            df['Broke_Out'] = 0
    
    n_breakouts = df['Broke_Out'].sum()
    print(f"   Found {n_breakouts} known breakout players ({n_breakouts/len(df):.1%})")
    
    if n_breakouts < 10:
        print("   Adding synthetic labels based on top performers...")
        # Label top 10% by a composite score as "breakouts"
        score_cols = ['SCA90', 'GCA90', 'xG_per_90', 'PrgC_per_90']
        available_score_cols = [c for c in score_cols if c in df.columns]
        if available_score_cols:
            df['temp_score'] = df[available_score_cols].fillna(0).mean(axis=1)
            threshold = df['temp_score'].quantile(0.9)
            df.loc[df['temp_score'] >= threshold, 'Broke_Out'] = 1
            df = df.drop(columns=['temp_score'])
            print(f"   Total breakouts after synthetic: {df['Broke_Out'].sum()}")
    
    # Train models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    results = {}
    for position in ['forward', 'midfielder', 'defender']:
        print(f"\n🎯 Training {position} model...")
        result = train_position_model(df, position)
        if result:
            results[position] = result
    
    # Print results
    print("\n" + "="*60)
    print("EMPIRICAL WEIGHTS RESULTS")
    print("="*60)
    
    for position, data in results.items():
        print(f"\n{'─'*40}")
        print(f"📊 {position.upper()}")
        print(f"{'─'*40}")
        print(f"   Samples: {data['n_samples']} | Breakouts: {data['n_breakouts']}")
        print(f"   CV AUC: {data['auc']:.3f}")
        print(f"\n   Weights:")
        for feature, weight in sorted(data['weights'].items(), key=lambda x: -abs(x[1])):
            direction = "↑" if weight > 0 else "↓"
            print(f"      {direction} {feature}: {weight:+.4f}")
    
    # Generate code
    print("\n" + "="*60)
    print("GENERATED CODE FOR PIPELINE")
    print("="*60)
    
    print("\nEMPIRICAL_WEIGHTS = {")
    for position, data in results.items():
        print(f"    '{position}': {{")
        for feature, weight in data['weights'].items():
            print(f"        '{feature}': {weight:.4f},")
        print("    },")
    print("}")
    
    # Save
    output_path = os.path.join(base_dir, 'empirical_weights.py')
    with open(output_path, 'w') as f:
        f.write("# Empirically derived weights\n")
        f.write("EMPIRICAL_WEIGHTS = {\n")
        for position, data in results.items():
            f.write(f"    '{position}': {{\n")
            for feature, weight in data['weights'].items():
                f.write(f"        '{feature}': {weight:.4f},\n")
            f.write("    },\n")
        f.write("}\n")
    
    print(f"\n💾 Weights saved to: {output_path}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
