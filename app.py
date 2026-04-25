import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import gzip
import io
import time
import scrython
import os


st.set_page_config(
    page_title="Magic: The Gathering Draft Synergy Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
    <style>
        /* This removes the top padding from the main content area */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        
        /* Make links white */
        a {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("MTG Draft Synergy Finder")
st.markdown("*Companion web app for finding new draft synergies with [17lands public data sets](https://www.17lands.com/public_datasets)*")

# Sidebar for parameters
st.sidebar.header("Configuration")

# Set selection
selected_set = st.sidebar.text_input(
    "Enter Set Code (3 letters)",
    value="",
    max_chars=3,
    help="Enter a 3-letter set code (KHM onwards)"
).upper()

@st.cache_data(show_spinner=False)
def get_cards_by_rarity(set_code):
    rarities = ['common', 'uncommon', 'rare', 'mythic']
    results = {}
    try:
        for r in rarities:
            search = scrython.cards.Search(q=f's:{set_code} r:{r} -is:digital')
            names = [card.name.split(' // ')[0] for card in search.data]
            # Remove non-English characters and normalize
            cleaned_names = []
            for name in names:
                # Keep only ASCII characters (English letters, numbers, basic punctuation)
                ascii_name = ''.join([c if ord(c) < 128 else '?' for c in name])
                if ascii_name:
                    cleaned_names.append(ascii_name)
            results[r] = cleaned_names
            time.sleep(0.1) 
    except scrython.base.ScryfallError as e:
        st.error(f"Error fetching cards for set \"{set_code}\": {str(e)}")
    return results

all_cards = []
non_rare_cards = []
valid_set_code = False

if selected_set:
    cards_by_rarity = get_cards_by_rarity(selected_set)
    valid_set_code = len(cards_by_rarity) > 0

if valid_set_code:
    cards_by_rarity['common'] = [card for card in cards_by_rarity['common'] if card not in ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest']]
    all_cards = cards_by_rarity['common'] + cards_by_rarity['uncommon'] + cards_by_rarity['rare'] + cards_by_rarity['mythic']
    non_rare_cards = cards_by_rarity['common'] + cards_by_rarity['uncommon']

# Parameters
synergy_card = st.sidebar.selectbox(
    "Select Synergy Card",
    options=all_cards,
    index=all_cards.index('Ominous Roost') if 'Ominous Roost' in all_cards else 0,
    help="Choose a card to look at synergies with"
)

synergy_number = st.sidebar.slider(
    "Number of Synergy Cards in Deck",
    min_value=1,
    max_value=5,
    value=1,
    help="Filter games with at least this many copies of the synergy card"
)

min_sample_size = st.sidebar.slider(
    "Minimum Sample Size",
    min_value=50,
    max_value=2000,
    value=100,
    step=10,
    help="Minimum games needed for a card be shown in the results"
)

# Function to get CSV row count for slider max value
@st.cache_data(show_spinner=False)
def get_csv_row_count(selected_set):
    """Get the total number of rows in the CSV file"""
    
    csv_filename = f"game_data_public.{selected_set}.PremierDraft.csv"
    
    # Check if local file exists
    if os.path.exists(csv_filename):
        try:
            # Quick way to get row count without loading full data
            with open(csv_filename, 'r') as f:
                row_count = sum(1 for _ in f) - 1  # Subtract header row
            return row_count
        except:
            pass
    return 1000000  # Fallback to large default

# Get row count if set is provided
if selected_set and len(selected_set) == 3:
    total_rows = get_csv_row_count(selected_set)
    max_games = st.sidebar.slider(
        "Maximum Games to Analyze",
        min_value=1000,
        max_value=total_rows,
        value=min(30000, total_rows),
        step=max(1000, total_rows // 100000 * 1000),
        help=f"Limit the number of games for faster processing (max: {total_rows:,} rows)"
    )
else:
    max_games = 30000  # Default value when no set is selected


@st.cache_data(show_spinner=False)
def get_card_image_url(card_name):
    """Generate Scryfall image URL for a card"""
    # Replace card name with URL-safe format
    card_name_safe = card_name.replace("'", "").replace(" ", "+").replace("-", "+").replace(",", "")
    return f"https://api.scryfall.com/cards/named?exact={card_name_safe}&format=image&version=normal"

def get_chunk_counts(chunk, cards, synergy_card, synergy_number):
    """Calculates raw counts (won/played) for a single chunk."""
    # Subset for Synergy
    deck_col = f"deck_{synergy_card}"
    has_synergy = chunk[deck_col] >= synergy_number
    chunk_syn = chunk[has_synergy]
    
    oh_cols = [f"opening_hand_{name}" for name in cards]
    dr_cols = [f"drawn_{name}" for name in cards]
    
    # helper to get counts from a dataframe
    def get_stats(df_subset):
        if df_subset.empty:
            return np.zeros(len(cards)), np.zeros(len(cards))
        # GIH Matrix: 1 if card was in opening hand OR drawn
        gih_matrix = (df_subset[oh_cols].values + df_subset[dr_cols].values > 0)
        counts = gih_matrix.sum(axis=0)
        wins = gih_matrix.T @ df_subset['won'].astype(np.float64).values
        return counts, wins

    gih_c, gih_w = get_stats(chunk)
    syn_c, syn_w = get_stats(chunk_syn)
    
    return gih_c, gih_w, syn_c, syn_w

def create_plotly_plot(plotdf, synergy_card, synergy_number):
    """Create interactive Plotly scatter plot"""
    
    # Create scatter plot
    fig = px.scatter(
        plotdf,
        x='GIH_wr',
        y='GIH_wr_syn',
        size=np.log(plotdf['n_GIH_syn']),
        color='n_GIH_syn',
        color_continuous_scale='agsunset',

        labels={
            'GIH_wr': 'Baseline Game-in-hand Winrate',
            'GIH_wr_syn': f'Game-in-hand Winrate with {synergy_number}x {synergy_card} in deck'
        },
        title=f'Win Rate Improvement with {synergy_number}x {synergy_card} in Deck'
    )
    
    # Add card names to hover data
    fig.update_traces(
        hovertemplate="<b>%{text}</b><br>",
        text=plotdf.index,
        unselected={'marker': {'opacity': 0.3}},
        selected={'marker': {'opacity': 1.0}}
    )

    fig.update_coloraxes(showscale=False)
    
    # Add diagonal line
    min_val = min(plotdf['GIH_wr'].min(), plotdf['GIH_wr_syn'].min())
    max_val = max(plotdf['GIH_wr'].max(), plotdf['GIH_wr_syn'].max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            hoverinfo='none',
            name='No Improvement Line',
            showlegend=True
        )
    )

    # Add average improvement line
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val + avg_improvement, max_val + avg_improvement],
            mode='lines',
            line=dict(color='darkred', dash='dot'),
            hoverinfo='none',
            name=f'Average Improvement: {avg_improvement*100:.1f}%',
            showlegend=True
        )
    )
    
    # Update layout
    fig.update_layout(
        width=800,
        height=700,
        xaxis=dict(autorange=True, tickmode='linear', tick0=0.4, dtick=0.05),
        yaxis=dict(autorange=True, tickmode='linear', tick0=0.4, dtick=0.05),
        legend=dict(orientation="v", yanchor="bottom", y=0.05, xanchor="right", x=1.0),
        showlegend=True
    )
    
    return fig

def get_needed_columns(csv_filename):
    """Peek at the header to determine which columns to keep."""
    needed_prefixes = ('deck_', 'opening_hand_', 'drawn_', 'won', 'rank', 'colors')
    header = pd.read_csv(csv_filename, nrows=0)
    return [c for c in header.columns if c.startswith(needed_prefixes)]

def download_and_save_raw(url, csv_filename):
    """Streams the download and decompresses directly to disk."""
    with st.spinner("Downloading and decompressing to disk..."):
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Decompress on the fly and write to local file
        with gzip.GzipFile(fileobj=response.raw) as gz:
            with open(csv_filename, 'wb') as f:
                for chunk in iter(lambda: gz.read(1024 * 1024), b''): # 1MB chunks
                    f.write(chunk)

@st.cache_data(show_spinner=False)
def analyze_large_csv(selected_set, max_games, synergy_card, synergy_number, min_sample_size):
    csv_filename = f"game_data_public.{selected_set}.PremierDraft.csv"

    if not os.path.exists(csv_filename):
        url = f"https://17lands-public.s3.amazonaws.com/analysis_data/game_data/{csv_filename}.gz"
        download_and_save_raw(url, csv_filename)
    
    cols_to_use = get_needed_columns(csv_filename)
    cards = [c.replace('opening_hand_', '') for c in cols_to_use if c.startswith('opening_hand_')]
    cards = [c for c in cards if c != synergy_card]
    
    # Initialize running totals
    total_gih_count = np.zeros(len(cards))
    total_gih_wins = np.zeros(len(cards))
    total_syn_count = np.zeros(len(cards))
    total_syn_wins = np.zeros(len(cards))

    dtype_dict = {col: 'int8' for col in cols_to_use if 'won' in col or 'deck_' in col}
    for col in cols_to_use:
        if 'opening_hand' in col or 'drawn' in col:
            dtype_dict[col] = 'int8'

    rows_processed = 0
    with pd.read_csv(csv_filename, usecols=cols_to_use, chunksize=10000, dtype=dtype_dict) as reader:
        for chunk in reader:
            # Respect max_games limit
            if rows_processed >= max_games:
                break
            
            # If chunk is larger than remaining limit, trim it
            if rows_processed + len(chunk) > max_games:
                chunk = chunk.head(max_games - rows_processed)
            
            # Get counts for this chunk
            gih_c, gih_w, syn_c, syn_w = get_chunk_counts(chunk, cards, synergy_card, synergy_number)
            
            # Accumulate
            total_gih_count += gih_c
            total_gih_wins += gih_w
            total_syn_count += syn_c
            total_syn_wins += syn_w
            
            rows_processed += len(chunk)

    # Final Win Rate Calculations (Vectorized)
    gih_wr = np.divide(total_gih_wins, total_gih_count, out=np.zeros_like(total_gih_wins), where=total_gih_count!=0)
    gih_wr_syn = np.divide(total_syn_wins, total_syn_count, out=np.zeros_like(total_syn_wins), where=total_syn_count!=0)
    
    plotdf = pd.DataFrame({
        'GIH_wr': gih_wr,
        'GIH_wr_syn': gih_wr_syn,
        'Improvement': (gih_wr_syn - gih_wr).round(3),
        'n_GIH_syn': total_syn_count
    }, index=cards)

    # 5. Calculate average improvement, weighted by sample size
    avg_improvement = (plotdf['Improvement'] * plotdf['n_GIH_syn']).sum() / plotdf['n_GIH_syn'].sum()
    
    # 6. Filter by sample size at the end (much faster)
    plotdf = plotdf[plotdf['n_GIH_syn'] > min_sample_size].dropna()

    return plotdf, avg_improvement

# Main content
# Only proceed if a set code is provided
if selected_set and len(selected_set) == 3 and valid_set_code:
    try:            
        # Analyze synergies
        with st.spinner("Analyzing synergies..."):
            plotdf, avg_improvement = analyze_large_csv(selected_set, max_games, synergy_card, synergy_number, min_sample_size)
        
        if len(plotdf) > 0:
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Interactive Plot", "Data Table"])
            
            with tab1:
                # Create columns for plot and card image
                plot_col, image_col = st.columns([3, 1])
                selected_card = None
                
                with plot_col:
                    fig = create_plotly_plot(plotdf, synergy_card, synergy_number)
                    event_data =st.plotly_chart(fig, width='stretch', on_select='rerun', key='scatter')

                    selected_points = event_data["selection"]["point_indices"]
                    if selected_points:
                        point_idx = selected_points[0]
                        actual_index = plotdf.index[point_idx]
                        selected_card = actual_index
                        card_data = plotdf.loc[selected_card]
                        improvement = card_data['Improvement'] * 100
                        description = "much better" if improvement > 6 \
                            else "better" if improvement > 2 \
                            else "slightly better" if improvement > 0 \
                            else "slightly worse" if improvement > -2 \
                            else "worse" if improvement > -6 \
                            else "much worse"
                        st.markdown(f"##### ***{selected_card}*** performs {description} in decks containing ***{synergy_number}*** ***{synergy_card}***.")
                
                with image_col:
                    
                    # Display synergy card
                    synergy_url = get_card_image_url(synergy_card)
                    st.image(synergy_url, width='stretch')
                    
                    # Display selected card (if any)
                    if selected_card:
                        card_url = get_card_image_url(selected_card)
                        st.image(card_url, width='stretch')
                        
                        # Show card stats for selected card
                        card_data = plotdf.loc[selected_card]
                        improvement = card_data['Improvement'] * 100
                        
                if selected_card:
                    col2, col3, col4, col5 = st.columns(4)
                    with col2:
                        st.metric("Baseline WR", f"{card_data['GIH_wr']:.1%}")
                    with col3:
                        st.metric("Synergy WR", f"{card_data['GIH_wr_syn']:.1%}")
                    with col4:
                        st.metric("Improvement", f"{improvement:+.1f}%")
                    with col5:
                        st.metric("Sample Size", f"{card_data['n_GIH_syn']:.0f}")
            
            with tab2:
                # Display data table
                display_df = plotdf.copy()
                display_df = display_df[['GIH_wr', 'GIH_wr_syn', 'Improvement', 'n_GIH_syn']]
                display_df.columns = ['Baseline WR', 'Synergy WR', 'Improvement', 'Sample Size']
                display_df = display_df.sort_values(by='Improvement', ascending=False)
                
                st.dataframe(display_df, width='stretch')
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.info("⬅ Please enter a 3-letter set code to begin analysis")
    
    # Show sample of what the app does
    st.markdown("---")
    st.subheader("📊 What This App Does")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📝 How to use:**
        1. Enter a set code and data will be downloaded from 17lands.
        2. Select a card to examine its synergies.
        3. The game-in-hand win rate of other cards will be plotted, showing how much they improve when the selected card is in the deck.
        4. Click on any data point for more information.
        """)
    
    with col2:
        st.markdown("""
        **💭 Interpretation:**
        - Data points are coloured by sample size.
        - Cards above the white line are synergistic with the selected card.
        - Cards below the white line are anti-synergistic.
        - The red dotted line shows the average improvement (indicates overall strength of the selected card).
        - Apparent synergies can be due to individual card synergies, or due to finding the correct deck/archetype for a certain card.
        """)
    
# Footer
st.markdown("---")
st.markdown("*Based on the YouTube video: [Finding new draft synergies with 17lands and python!](https://www.youtube.com/watch?v=TvRQKlT0pN0)*")
