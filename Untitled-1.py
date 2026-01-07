"""
Football Superstar Prediction Dashboard
Interactive Streamlit app to explore predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime

# Remove default config, we set it in main now
# st.set_page_config(
#     page_title="Football Superstar Predictor",
#     page_icon="⚽",
#     layout="wide"
# )

# Load data
# Load data
def calculate_proprietary_metrics(df):
    """Calcs the Secret Metrics (Ghost, Alpha, Progression) for any dataframe."""
    df = df.copy()
    
    # Ensure columns exist (fill with 0 if missing for robustness)
    req_cols = ['PrgC', 'PrgP', 'Gls', 'xG', 'SCA90', 'Touches']
    for c in req_cols:
        if c not in df.columns:
            df[c] = 0
            
    # 1. The "Beast Mode" (Progression)
    df['Progression_Score'] = df['PrgC'] + df['PrgP']
    
    # 2. "Lethal Finisher" (xG Overperformance)
    df['Finishing_Alpha'] = df['Gls'] - df['xG']
    
    # 3. "Ghost Factor" (Threat per Touch)
    df['Ghost_Factor'] = df.apply(lambda x: x['SCA90'] / x['Touches'] * 100 if x['Touches'] > 0 else 0, axis=1)
    
    return df

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        df = pd.read_csv(os.path.join(base_dir, 'data/processed/final_predictions.csv'))
    except:
        df = pd.read_csv(os.path.join(base_dir, 'data/processed/predictions_advanced.csv'))
    
    # MERGE WITH RAW DATA FOR SECRET METRICS
    raw = load_raw_data()
    # Select useful columns from raw to avoid duplicates
    cols_to_use = ['Player', 'Squad', 'xG', 'npxG', 'SCA90', 'PrgC', 'PrgP', 'Gls', 'Touches']
    raw_subset = raw[cols_to_use].copy()
    
    # Merge
    merged = pd.merge(df, raw_subset, on=['Player', 'Squad'], how='left')
    
    # Apply Metrics
    merged = calculate_proprietary_metrics(merged)
    
    return merged

def find_similar_players(target_name, pool_df, features=['Ghost_Factor', 'Progression_Score']):
    """Finds players in pool_df statistically similar to target_name."""
    
    # Get target stats (assuming target_name is in pool_df, or we passed a single row DF as target)
    target_row = pool_df[pool_df['Player'] == target_name]
    
    if target_row.empty:
        return None
        
    # Scale Data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(pool_df[features].fillna(0))
    
    # Target Vector
    target_idx = pool_df.index[pool_df['Player'] == target_name].tolist()[0]
    target_vector = scaled_features[target_idx].reshape(1, -1)
    
    # Calc Similarity
    similarity = cosine_similarity(target_vector, scaled_features)
    
    # Add to DF
    pool_df = pool_df.copy()
    pool_df['Similarity'] = similarity[0]
    
    return pool_df.sort_values('Similarity', ascending=False)

@st.cache_data
def load_raw_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base_dir, 'data/raw/players_data-2024_2025.csv'))

# --- PORTFOLIO SYSTEM ---
def get_portfolio_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'data/processed/my_scouting_portfolio.csv')

def load_portfolio():
    path = get_portfolio_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=['Player', 'Club', 'DC_Index', 'Date_Scouted', 'Notes'])
    return pd.read_csv(path)

def add_to_portfolio(player_row, prob_col, note):
    portfolio = load_portfolio()
    
    # Check if player already exists
    if player_row['Player'] in portfolio['Player'].values:
        return False, "Player already in portfolio."
        
    new_entry = {
        'Player': player_row['Player'],
        'Club': player_row['Squad'],
        'DC_Index': player_row[prob_col],
        'Date_Scouted': datetime.now().strftime("%Y-%m-%d"),
        'Notes': note
    }
    
    portfolio = pd.concat([portfolio, pd.DataFrame([new_entry])], ignore_index=True)
    portfolio.to_csv(get_portfolio_path(), index=False)
    return True, f"✅ Scaled {player_row['Player']} into Portfolio!"

# Helper function for Player Card
def render_player_card(player, prob_col, intuition=0):
    base_score = int(player[prob_col] * 99)
    final_score = max(min(base_score + intuition, 99), 1) # Clamp between 1-99
    
    card_class = "platinum" if final_score >= 90 else "gold"
    
    # Map stats to generic FIFA-like attributes (simplified for demo)
    pac = int(player.get('Speed', 70) if 'Speed' in player else 70 + np.random.randint(-5, 5))
    sho = int(player['Goals_per_90'] * 50 + 50) if 'Goals_per_90' in player else 60
    pas = int(player['Assists_per_90'] * 50 + 50) if 'Assists_per_90' in player else 60
    dri = int(player.get('Dribbles', 75) if 'Dribbles' in player else 75)
    
    html = f"""
    <div class="player-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="text-align: left;">
                <div class="player-rating">{final_score}</div>
                <div class="player-pos">{player['Position_Group'][:3].upper()}</div>
            </div>
            <div style="font-size: 40px;">⚽</div>
        </div>
        <div class="player-name">{player['Player'].split()[-1]}</div>
        <div class="stat-grid">
            <div>{pac} PAC</div>
            <div>{dri} DRI</div>
            <div>{sho} SHO</div>
            <div>{25} DEF</div>
            <div>{pas} PAS</div>
            <div>{70} PHY</div>
        </div>
        <div style="margin-top: 10px; font-size: 12px; opacity: 0.8;">DC INDEX</div>
    </div>
    """
    return html

# Main app
def main():
    st.set_page_config(page_title="Conaghan Scout Pro", page_icon="⚽", layout="wide")
    
    # Custom CSS
    st.markdown("""
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
            color: #4ade80; /* Neon Green */
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
            border-radius: 20px; # Fixed radius for cleaner look
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
        </style>
    """, unsafe_allow_html=True)

    st.title("DC SCOUT PRO")
    st.markdown("### <span style='color: #4ade80'>Next-Gen Scouting Intelligence</span>", unsafe_allow_html=True)

    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run the notebook first to generate predictions.")
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    # Position filter
    positions = ['All'] + sorted(df['Position_Group'].dropna().unique().tolist())
    selected_pos = st.sidebar.selectbox("Position", positions)

    # Age filter
    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

    # Probability filter
    min_prob = st.sidebar.slider("Minimum Superstar Probability", 0, 100, 50) / 100
    
    # Intuition Override (The 0 to 1 Feature)
    st.sidebar.markdown("---")
    st.sidebar.header("Scout Override")
    intuition_score = st.sidebar.slider("Dale's Eye Test (Adjustment)", -20, 20, 0, help="Adjust the model score based on your domain knowledge.")
    
    # Apply filters
    filtered_df = df.copy()
    if selected_pos != 'All':
        filtered_df = filtered_df[filtered_df['Position_Group'] == selected_pos]
    filtered_df = filtered_df[
        (filtered_df['Age'] >= age_range[0]) &
        (filtered_df['Age'] <= age_range[1])
    ]

    prob_col = 'Ensemble_Probability' if 'Ensemble_Probability' in df.columns else 'XGB_Probability'
    if prob_col not in filtered_df.columns:
        prob_col = [c for c in filtered_df.columns if 'Probability' in c][0]

    # Portfolio Quick Add Logic (Moved here to access prob_col)
    st.sidebar.markdown("---")
    st.sidebar.header("Scout Notebook")
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

    filtered_df = filtered_df[filtered_df[prob_col] >= min_prob]

    # Main content - tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Top Prospects", "Analytics", "Player Comparison", "Full Data", "My Portfolio", "DNA Matcher"])

    with tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.header("The DC Index: Top Prospects")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Scouted Players", len(filtered_df))
        with col2:
            st.metric("Avg DC Score", f"{filtered_df[prob_col].mean()*100:.1f}")
        with col3:
            high_prob = (filtered_df[prob_col] > 0.9).sum()
            st.metric("Elite Tier (>90)", high_prob)
        with col4:
            avg_age = filtered_df['Age'].mean()
            st.metric("Avg Age", f"{avg_age:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Top 3 Cards
        st.subheader("Market Leaders")
        top3 = filtered_df.nlargest(3, prob_col)
        
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        
        for i, (idx, row) in enumerate(top3.iterrows()):
            with cols[i]:
                st.markdown(render_player_card(row, prob_col, intuition_score), unsafe_allow_html=True)

        st.markdown("---")

        # Top players table
        st.subheader("Scouting Shortlist")
        top20 = filtered_df.nlargest(20, prob_col)[['Player', 'Age', 'Squad', 'Position_Group', 'Goals_per_90', 'Assists_per_90', prob_col]]
        top20.columns = ['Player', 'Age', 'Club', 'Position', 'Goals/90', 'Assists/90', 'DC Index']
        st.dataframe(
            top20.style.background_gradient(subset=['DC Index'], cmap='Greens')
                 .format({'DC Index': '{:.1%}'}),
            use_container_width=True, hide_index=True
        )

    with tab2:
        st.header("Analytical Deep Dive")
        st.markdown("<div style='color: rgba(255,255,255,0.7); margin-bottom: 20px; font-style: italic;'>Advanced metrics visualization powered by Plotly.</div>", unsafe_allow_html=True)
        
        # Secret Metrics Row
        st.markdown("### Proprietary 'Secret' Intelligence")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Top Finisher (Alpha)", 
                     filtered_df.sort_values('Finishing_Alpha', ascending=False).iloc[0]['Player'],
                     f"+{filtered_df['Finishing_Alpha'].max():.2f} xG")
        with c2:
            st.metric("Most Progressive", 
                     filtered_df.sort_values('Progression_Score', ascending=False).iloc[0]['Player'],
                     f"{filtered_df['Progression_Score'].max():.0f} Acts")
        with c3:
            st.metric("Ghost Factor (Efficiency)", 
                     filtered_df.sort_values('Ghost_Factor', ascending=False).iloc[0]['Player'],
                     f"{filtered_df['Ghost_Factor'].max():.2f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            # Finishing Alpha vs DC Index
            st.caption("System vs Killer?")
            with st.expander("How to read this"):
                st.markdown("""
                *   **Top Right:** High Index + High Finishing Alpha = **Lethal Finishers** (The Goal Machines).
                *   **Bottom Right:** High Index + Negative Alpha = **Underperformers** (Getting chances, missing them).
                """)
            fig = px.scatter(
                filtered_df, x=prob_col, y='Finishing_Alpha', 
                color='Position_Group', size='Progression_Score', hover_name='Player',
                title='Value Finding: High Index + High Finishing Alpha', template='plotly_dark'
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Underperforming xG")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Ghost Factor vs Touches
            st.caption("The Haaland Scale")
            with st.expander("How to read this"):
                st.markdown("""
                *   **Top Left (The Ghost):** Low Touches + High Danger (SCA). Efficient assassins.
                *   **Bottom Right (Passenger):** High Touches + Low Danger. Stat-padders.
                """)
            fig = px.scatter(
                filtered_df, x='Touches', y='SCA90',
                color='Ghost_Factor', size=prob_col, hover_name='Player',
                title='The "Ghost" Chart: High Danger, Low Touches', template='plotly_dark',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Bias Check")
            with st.expander("How to read this"):
                st.markdown("If Forwards (FW) are much higher than Defenders (DF), mentally adjust your expectations. A Defender with 0.5 might be Elite for their role.")
            pos_avg = df.groupby('Position_Group')[prob_col].mean().reset_index()
            fig = px.bar(pos_avg, x='Position_Group', y=prob_col, 
                         title='Index by Position', color=prob_col, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.caption("Hidden Gem Finder")
            with st.expander("How to read this"):
                st.markdown("**Look for Bottom-Left / Dark Green:** Players with LOW goals/assists but HIGH DC Index. These are the players the model sees potential in *before* they output stats.")
            fig = px.scatter(
                filtered_df, x='Goals_per_90', y='Assists_per_90', 
                color=prob_col, size=prob_col, hover_name='Player',
                title='Goal Contributions vs Index', template='plotly_dark',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Head-to-Head Analysis")
        
        c1, c2 = st.columns([1, 1])
        players = filtered_df['Player'].tolist()
        
        with c1:
            p1_name = st.selectbox("Select Target", players, index=0)
        with c2:
            p2_name = st.selectbox("Select Benchmark", players, index=min(1, len(players)-1))
            
        if p1_name and p2_name:
            p1 = filtered_df[filtered_df['Player'] == p1_name].iloc[0]
            p2 = filtered_df[filtered_df['Player'] == p2_name].iloc[0]
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(render_player_card(p1, prob_col, intuition_score), unsafe_allow_html=True)
            with c2:
                st.markdown(render_player_card(p2, prob_col, intuition_score), unsafe_allow_html=True)
                
            # Radar
            categories = ['Goals/90', 'Assists/90', 'DC Index']
            p1_vals = [p1['Goals_per_90']*100, p1['Assists_per_90']*100, p1[prob_col]*100]
            p2_vals = [p2['Goals_per_90']*100, p2['Assists_per_90']*100, p2[prob_col]*100]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=p1_vals + [p1_vals[0]], theta=categories + [categories[0]], fill='toself', name=p1_name, line_color='#4ade80'))
            fig.add_trace(go.Scatterpolar(r=p2_vals + [p2_vals[0]], theta=categories + [categories[0]], fill='toself', name=p2_name, line_color='#A0A0A0'))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False), bgcolor='rgba(0,0,0,0)'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title="Skillset Overlap"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with tab4:
        st.header("Raw Intelligence Data")
        st.dataframe(filtered_df.sort_values(prob_col, ascending=False), use_container_width=True)

    with tab5:
        st.header("My Scouting Portfolio")
        portfolio_df = load_portfolio()
        
        if portfolio_df.empty:
            st.info("Your portfolio is empty. Go to the Sidebar to add players!")
        else:
            # Portfolio Metrics
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total Scouted", len(portfolio_df))
            with c2:
                # Ensure it operates on numeric data
                avg_score = pd.to_numeric(portfolio_df['DC_Index'], errors='coerce').mean()
                st.metric("Portfolio Quality (Avg)", f"{avg_score * 100:.1f}")
            
            # Display Portfolio
            st.dataframe(
                portfolio_df.style.background_gradient(subset=['DC_Index'], cmap='Greens')
                            .format({'DC_Index': '{:.1%}'}),
                use_container_width=True,
                hide_index=True
            )
            
            # Download
            csv = portfolio_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Export Report",
                csv,
                "dc_scout_portfolio.csv",
                "text/csv",
                key='download-csv'
            )

    with tab6:
        st.header("DNA Matcher: Find Replacements")
        st.markdown("> **Scenario:** You lost your star. Find a clone.")
        
        # Load FULL raw data for the search target (so we can find Mitoma even if he is old)
        try:
            raw_all = load_raw_data()
            raw_all = calculate_proprietary_metrics(raw_all)
            
            # Fix: Ensure Position_Group exists
            if 'Position_Group' not in raw_all.columns and 'Pos' in raw_all.columns:
                 raw_all['Position_Group'] = raw_all['Pos'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)

            all_players_search = sorted(raw_all['Player'].unique().tolist())
        except:
             # Fallback to U23 list if raw fails
            raw_all = filtered_df
            all_players_search = sorted(filtered_df['Player'].unique().tolist())

        st.markdown("---")
        st.markdown("### Step 1: Select the Star to Replace")
        st.caption("Search for any player in the Top 5 Leagues (even if they are over 23). We will find their younger clone.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            target_name = st.selectbox("Search Player Database:", all_players_search, index=0, label_visibility="collapsed")
        
        st.markdown("---")
        
        st.markdown("### Step 2: Run the Algorithm")
        if st.button("SCAN DATABASE FOR MATCHES", type="primary", use_container_width=True):
            # 1. Get Target Profile
            # Note: We must fetch the target from 'raw_all' because 'filtered_df' only has U23s
            target_profile = raw_all[raw_all['Player'] == target_name]
            
            if target_profile.empty:
                st.error("Player not found in database.")
            else:
                # 2. Run Similarity Search against the U23 PROSPECTS (filtered_df)
                # We need to temporarily add the TARGET to the POOL if they aren't in it, to make sure scaling works nicely,
                # BUT `find_similar_players` handles single-row targets? 
                # Actually, `find_similar_players` expects the target to be IN the pool to find its index.
                # So we will Append target to pool, calc similarity, then remove target.
                
                pool = filtered_df.copy()
                
                # If target is NOT in pool (e.g. Mitoma is 26, Pool is U23), add him temporarily
                if target_name not in pool['Player'].values:
                    # We need to ensure columns match. 
                    # raw_all might have different cols than filtered_df (which has predictions).
                    # filtered_df has 'Ensemble_Probability'. raw_all does not.
                    # matches need proprietary metrics.
                    
                    # Align columns for the pool
                    cols_needed = ['Player', 'Squad', 'Position_Group', 'Ghost_Factor', 'Progression_Score', 'Finishing_Alpha']
                    
                    # Create a standard pool with just needed cols for matching
                    search_pool = pool[cols_needed].copy()
                    
                    # Prepare target row
                    target_row_std = target_profile[cols_needed].copy()
                    
                    # Combine
                    search_pool = pd.concat([search_pool, target_row_std], ignore_index=True)
                else:
                    search_pool = pool[['Player', 'Squad', 'Position_Group', 'Ghost_Factor', 'Progression_Score', 'Finishing_Alpha']].copy()

                # Run Matcher
                # We use Ghost Factor, Progression, and Alpha as the 'DNA'
                results = find_similar_players(target_name, search_pool, features=['Ghost_Factor', 'Progression_Score', 'Finishing_Alpha'])
                
                # Remove self-match
                results = results[results['Player'] != target_name]
                
                # Filter by Position (Optional but recommended)
                target_pos = target_profile.iloc[0]['Position_Group']
                results = results[results['Position_Group'] == target_pos]
                
                # Show Top 3
                st.success(f"Found {len(results)} matches for {target_name} ({target_pos})")
                
                top_matches = results.head(3)
                
                c1, c2, c3 = st.columns(3)
                cols = [c1, c2, c3]
                
                # We need to fetch the FULL data for these Top 3 from 'filtered_df' to render cards
                # because 'results' only has the search columns.
                
                for i, (idx, row) in enumerate(top_matches.iterrows()):
                    # Find this player in our main filtered_df to get their DC Index
                    original_row = filtered_df[filtered_df['Player'] == row['Player']]
                    if not original_row.empty:
                        original_row = original_row.iloc[0]
                        with cols[i]:
                            st.caption(f"Match Quality: {row['Similarity']*100:.1f}%")
                            st.markdown(render_player_card(original_row, prob_col, intuition_score), unsafe_allow_html=True)

if __name__ == "__main__":
    main()