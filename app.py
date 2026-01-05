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

# Page config
st.set_page_config(
    page_title="Football Superstar Predictor",
    page_icon="⚽",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/final_predictions.csv')
    except:
        df = pd.read_csv('data/processed/predictions_advanced.csv')
    return df

@st.cache_data
def load_raw_data():
    return pd.read_csv('data/raw/players_data-2024_2025.csv')

# Main app
def main():
    st.title("⚽ Football Superstar Prediction")
    st.markdown("*Predicting which young players (18-24) will become superstars*")

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

    filtered_df = filtered_df[filtered_df[prob_col] >= min_prob]

    # Main content - tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top Prospects", "📊 Analytics", "🔍 Player Comparison", "📋 Full Data"])

    with tab1:
        st.header("Top Superstar Prospects")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Players", len(filtered_df))
        with col2:
            st.metric("Avg Probability", f"{filtered_df[prob_col].mean():.0%}")
        with col3:
            high_prob = (filtered_df[prob_col] > 0.9).sum()
            st.metric("High Probability (>90%)", high_prob)
        with col4:
            avg_age = filtered_df['Age'].mean()
            st.metric("Average Age", f"{avg_age:.1f}")

        # Top players table
        st.subheader("Top 20 Prospects")
        top20 = filtered_df.nlargest(20, prob_col)[['Player', 'Age', 'Squad', 'Position_Group', 'Goals_per_90', 'Assists_per_90', prob_col]]
        top20.columns = ['Player', 'Age', 'Club', 'Position', 'Goals/90', 'Assists/90', 'Superstar Prob']
        top20['Superstar Prob'] = top20['Superstar Prob'].apply(lambda x: f"{x:.0%}")
        st.dataframe(top20, use_container_width=True, hide_index=True)

        # Bar chart
        top10 = filtered_df.nlargest(10, prob_col)
        fig = px.bar(
            top10,
            x=prob_col,
            y='Player',
            orientation='h',
            title='Top 10 Superstar Probabilities',
            color=prob_col,
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Analytics Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            # Distribution by position
            pos_avg = df.groupby('Position_Group')[prob_col].mean().reset_index()
            fig = px.bar(
                pos_avg,
                x='Position_Group',
                y=prob_col,
                title='Average Superstar Probability by Position',
                color=prob_col,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Age distribution
            fig = px.histogram(
                filtered_df,
                x='Age',
                color='Position_Group',
                title='Age Distribution of Prospects',
                nbins=7
            )
            st.plotly_chart(fig, use_container_width=True)

        # Scatter plot: Goals vs Assists
        fig = px.scatter(
            filtered_df,
            x='Goals_per_90',
            y='Assists_per_90',
            color=prob_col,
            size=prob_col,
            hover_name='Player',
            hover_data=['Squad', 'Age'],
            title='Goals vs Assists (per 90 minutes)',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Probability distribution
        fig = px.histogram(
            filtered_df,
            x=prob_col,
            nbins=20,
            title='Distribution of Superstar Probabilities'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Player Comparison")

        col1, col2 = st.columns(2)

        players = filtered_df['Player'].tolist()

        with col1:
            player1 = st.selectbox("Select Player 1", players, index=0)
        with col2:
            player2 = st.selectbox("Select Player 2", players, index=min(1, len(players)-1))

        if player1 and player2:
            p1 = filtered_df[filtered_df['Player'] == player1].iloc[0]
            p2 = filtered_df[filtered_df['Player'] == player2].iloc[0]

            # Comparison metrics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(player1)
                st.write(f"**Club:** {p1['Squad']}")
                st.write(f"**Age:** {int(p1['Age'])}")
                st.write(f"**Position:** {p1['Position_Group']}")
                st.metric("Superstar Probability", f"{p1[prob_col]:.0%}")
                st.metric("Goals/90", f"{p1['Goals_per_90']:.2f}")
                st.metric("Assists/90", f"{p1['Assists_per_90']:.2f}")

            with col2:
                st.subheader(player2)
                st.write(f"**Club:** {p2['Squad']}")
                st.write(f"**Age:** {int(p2['Age'])}")
                st.write(f"**Position:** {p2['Position_Group']}")
                st.metric("Superstar Probability", f"{p2[prob_col]:.0%}")
                st.metric("Goals/90", f"{p2['Goals_per_90']:.2f}")
                st.metric("Assists/90", f"{p2['Assists_per_90']:.2f}")

            # Radar chart comparison
            categories = ['Goals/90', 'Assists/90', 'Superstar Prob']

            # Normalize values for radar
            p1_vals = [p1['Goals_per_90']*100, p1['Assists_per_90']*100, p1[prob_col]*100]
            p2_vals = [p2['Goals_per_90']*100, p2['Assists_per_90']*100, p2[prob_col]*100]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=p1_vals + [p1_vals[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=player1
            ))
            fig.add_trace(go.Scatterpolar(
                r=p2_vals + [p2_vals[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=player2
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Player Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Full Dataset")

        # Search
        search = st.text_input("Search player name")
        if search:
            display_df = filtered_df[filtered_df['Player'].str.contains(search, case=False, na=False)]
        else:
            display_df = filtered_df

        # Display
        st.dataframe(
            display_df.sort_values(prob_col, ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="superstar_predictions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
