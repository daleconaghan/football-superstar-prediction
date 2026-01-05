"""
Download football player statistics from FBref (free, no API key needed).
This script fetches the Big 5 European leagues player stats for 2024-25 season.
"""

import pandas as pd
import time

def download_fbref_data():
    """
    Download player stats from FBref for the Big 5 European leagues.
    Returns a DataFrame with player statistics.
    """

    # FBref URLs for 2024-25 Big 5 European Leagues standard stats
    # These are public, no authentication needed
    urls = {
        'standard': 'https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats',
        'shooting': 'https://fbref.com/en/comps/Big5/shooting/players/Big-5-European-Leagues-Stats',
        'passing': 'https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats',
        'defense': 'https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats',
    }

    print("Downloading player data from FBref...")
    print("This may take a minute (respecting rate limits)...\n")

    try:
        # Download standard stats (main table)
        print("Fetching standard stats...")
        tables = pd.read_html(urls['standard'])

        # The player stats table is usually the first one with many columns
        df = None
        for table in tables:
            if len(table.columns) > 20:  # Player stats table has many columns
                df = table
                break

        if df is None:
            print("Could not find player stats table")
            return None

        # Clean up multi-level column headers if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns]

        # Remove rows that are header repeats
        df = df[df.iloc[:, 0] != df.columns[0]]

        # Reset index
        df = df.reset_index(drop=True)

        print(f"Downloaded {len(df)} players")
        print(f"Columns: {len(df.columns)}")

        return df

    except Exception as e:
        print(f"Error downloading data: {e}")
        print("\nTrying alternative method...")
        return None


def download_sample_data():
    """
    If FBref fails, create sample data structure that matches what we need.
    You can then manually download from Kaggle and place in data/raw/
    """

    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("""
To get the full dataset, please:

1. Go to: https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025

2. Click 'Download' (you'll need a free Kaggle account)

3. Extract the CSV file and place it in:
   ~/football-superstar-prediction/data/raw/

4. Rename it to: players_data_2024_2025.csv

Alternative: You can also use FBref directly:
- Go to https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats
- Click "Share & Export" -> "Get table as CSV"
- Save to ~/football-superstar-prediction/data/raw/

""")
    print("="*60)


if __name__ == "__main__":
    df = download_fbref_data()

    if df is not None:
        # Save the data
        output_path = "../data/raw/players_data_2024_2025.csv"
        df.to_csv(output_path, index=False)
        print(f"\nData saved to: {output_path}")
        print(f"\nFirst few rows:")
        print(df.head())
    else:
        download_sample_data()
