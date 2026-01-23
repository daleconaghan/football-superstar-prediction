# DC Pro: Next-Gen Football Scouting Intelligence

A data-driven talent identification system that uses machine learning to predict which young footballers are most likely to become elite players.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Live Demo

**[View Live Application →](https://dc-football-scout.streamlit.app/)**


## Overview

DC Pro analyses performance data from Europe's top 5 leagues to identify high-potential U24 players before they break out. Unlike traditional scouting metrics that focus on goals and assists, the **DC Index** uses empirically-derived weights trained on historical breakout data.

### Key Features

- **DC Index** — Proprietary potential score based on metrics that actually predict breakouts
- **Market Valuations** — Multi-factor estimates incorporating age, position, league, and form
- **DNA Matcher** — Find statistical clones of any player
- **Head-to-Head Analysis** — Compare prospects side-by-side with radar charts
- **Portfolio Tracker** — Build and monitor a shortlist of targets

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/daleconaghan/football-superstar-prediction.git
cd football-superstar-prediction

# Install dependencies
pip install -r requirements.txt

# Run the pipeline to generate predictions
python pipeline.py

# Launch the dashboard
streamlit run football.py
```

### Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.3.0
```

## Project Structure

```
football-superstar-prediction/
├── football.py                 # Streamlit dashboard (main app)
├── pipeline.py                 # Data processing and DC Index calculation
├── train_weights_simple.py     # Empirical weight training script
├── empirical_weights.py        # Trained model coefficients
├── data/
│   ├── raw/                    # FBRef source data (2021-2026)
│   └── processed/              # Pipeline output
├── requirements.txt
└── README.md
```

## Methodology

### Data Sources

- **Source:** FBRef.com (via StatsBomb)
- **Coverage:** Premier League, La Liga, Bundesliga, Serie A, Ligue 1
- **Seasons:** 2021-22 through 2025-26
- **Sample:** 14,229 player-seasons; 1,599 U23 players with 500+ minutes

### DC Index Training

The DC Index weights were derived using logistic regression on historical data:

1. Filtered to U23 players with 500+ minutes played
2. Labelled known "breakout" players (Bellingham, Saka, Musiala, Pedri, etc.)
3. Trained position-specific models to identify predictive features
4. Extracted coefficients as empirical weights

### Learned Weights

| Position | Top Predictors |
|----------|----------------|
| **Forward** | Progressive Carries (+2.21), xAG (+1.90), xG (+1.73) |
| **Midfielder** | Progressive Passes (+1.19), Progressive Carries (+0.82), Key Passes (+0.61) |
| **Defender** | Progressive Passes (+2.00), Tackles (+1.52), Age (-0.52) |

**Key Finding:** Progressive ball movement is the strongest predictor across all positions.

### Market Valuation Model

Estimated values incorporate six factors:

| Factor | Impact |
|--------|--------|
| DC Index | Base potential score |
| Age | U19: +50%, U22: +20%, 25+: -20% |
| League | Premier League: +50%, Top 4 leagues: +20% |
| Position | Forward: +25%, Midfielder: +10%, Defender: -10%, GK: -30% |
| Minutes | 2500+: +10%, <500: -40% (sample size confidence) |
| Form | Outperforming xG: up to +15% |

### Proprietary Metrics

- **Finishing Alpha:** Goals − xG (clinical finishing ability)
- **Ghost Factor:** SCA / Touches × 100 (efficiency per touch)
- **Progression Score:** (PrgC + PrgP) / 90s (ball advancement)

## Usage

### Running the Dashboard

```bash
streamlit run football.py
```

Navigate to `http://localhost:8501` in your browser.

### Regenerating Predictions

To update predictions with new data:

```bash
# Place new FBRef data in data/raw/
python pipeline.py
```

### Retraining Weights

To retrain the DC Index weights on updated historical data:

```bash
python train_weights_simple.py
```

## Limitations

- No contract length data (significantly affects real transfer fees)
- Limited goalkeeper metrics in source data
- No injury history or fitness data
- Market valuations are heuristic, not trained on actual fees
- In-sample training — true predictive accuracy would require out-of-sample validation

## Future Work

- [ ] Scrape Transfermarkt for actual transfer fees to train valuation model
- [ ] Add contract expiry data
- [ ] Incorporate season-over-season trajectory
- [ ] Expand to more leagues (Eredivisie, Liga Portugal, Championship)
- ~~Deploy to Streamlit Cloud~~ ✅ **DEPLOYED**
  
## Author

**Dale Conaghan**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FBRef](https://fbref.com/) for comprehensive football statistics
- [StatsBomb](https://statsbomb.com/) for advanced metrics
- [Streamlit](https://streamlit.io/) for the dashboard framework
