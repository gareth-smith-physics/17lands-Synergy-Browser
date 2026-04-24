# MTG Draft Synergy Analyzer

A Streamlit web application for analyzing card synergies in Magic: The Gathering draft data from 17lands.

## Features

- 🎯 Analyze card synergies in draft decks
- 📈 Interactive win rate comparisons using Plotly
- 🔍 Filter by sample size and synergy count
- 📊 Sort by biggest improvements
- 📁 Downloads data from 17lands

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Enter a 17lands set code
2. Select a synergy card from the dropdown
3. Adjust parameters (synergy count, sample size, max games)
4. View the interactive plot showing win rate improvements

## Original Project

This web app is based on the Jupyter notebook from the YouTube video: [Finding new draft synergies with 17lands and python!](https://www.youtube.com/watch?v=TvRQKlT0pN0)
