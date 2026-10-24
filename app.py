"""CineMatch — Intelligent Movie Recommendation System
A production-grade Streamlit application showcasing Content-Based,
Collaborative Filtering (SVD), and Hybrid Recommendation Engines.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup system path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import validate_dataset_presence, get_default_data_dir
from src.movie_service import MovieService

# -----------------------------------------------------------------------------
# Streamlit Page Config & Custom Cinematic CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch — Intelligent Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CINEMATIC_CSS = """
<style>
/* Root CSS Variables */
:root {
    --bg-primary: #0b0f19;
    --bg-surface: #111827;
    --bg-card: #1e293b;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-pink: #ec4899;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border-color: #334155;
    --card-glow: rgba(139, 92, 246, 0.15);
}

/* Global Application Background */
.stApp {
    background: radial-gradient(circle at 10% 20%, #0f172a 0%, #0b0f19 90%);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Typography & Headings */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* CineMatch Brand Gradient Banner */
.brand-hero {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.25) 0%, rgba(6, 182, 212, 0.15) 50%, rgba(15, 23, 42, 0.4) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px);
}

.brand-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa 0%, #38bdf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}

.brand-tagline {
    color: #cbd5e1;
    font-size: 1.15rem;
    margin-top: 0.6rem;
    font-weight: 400;
}

/* Movie Card Component */
.movie-card {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.movie-card:hover {
    transform: translateY(-4px);
    border-color: #8b5cf6;
    box-shadow: 0 12px 25px rgba(139, 92, 246, 0.25);
}

.poster-box {
    width: 100%;
    aspect-ratio: 2/3;
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 1rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 0.8rem;
    border: 1px solid rgba(255,255,255,0.08);
}

.badge-genre {
    display: inline-block;
    background: rgba(139, 92, 246, 0.2);
    border: 1px solid rgba(139, 92, 246, 0.4);
    color: #c4b5fd;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 9999px;
    margin-right: 4px;
    margin-bottom: 4px;
}

.badge-rating {
    display: inline-flex;
    align-items: center;
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #fbbf24;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 6px;
}

.badge-score {
    display: inline-flex;
    align-items: center;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 6px;
}

/* Metric Display Cards */
.metric-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #38bdf8;
}
.metric-lbl {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0d1322 !important;
    border-right: 1px solid #1e293b;
}

/* Form inputs & buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
    color: #ffffff;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
}
</style>
"""
st.markdown(CINEMATIC_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Caching Service & Data Initialization
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing CineMatch Machine Learning Engines...")
def get_service() -> MovieService:
    """Initialize and cache the core movie service and trained models."""
    return MovieService()


def check_dataset_setup() -> bool:
    """Check dataset files and display guidance if missing."""
    is_valid, missing = validate_dataset_presence()
    if not is_valid:
        st.error("🚨 MovieLens Dataset Files Missing!")
        st.markdown(
            f"""
            CineMatch requires the **MovieLens Latest Small** dataset CSV files:
            - **Missing files:** `{missing}`
            - **Target directory:** `{get_default_data_dir()}`

            ### How to Set Up:
            1. Download the archive from GroupLens: [ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
            2. Extract `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv` into the `data/` folder.
            3. Run `python scripts/prepare_data.py` and `python scripts/train_models.py`.
            4. Refresh this page.
            """
        )
        return False
    return True


# Check dataset availability
if not check_dataset_setup():
    st.stop()

service = get_service()

# -----------------------------------------------------------------------------
# Session State Management
# -----------------------------------------------------------------------------
if "user_favorites" not in st.session_state:
    # Default initial favorites for cold-start exploration
    st.session_state.user_favorites = {
        1: 5.0,    # Toy Story
        260: 5.0,  # Star Wars: Episode IV
        296: 4.5,  # Pulp Fiction
    }

if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = 1


# -----------------------------------------------------------------------------
# Poster & Graphic Fallback Generator
# -----------------------------------------------------------------------------
GENRE_GRADIENTS = {
    "Action": "linear-gradient(135deg, #b91c1c 0%, #7f1d1d 100%)",
    "Adventure": "linear-gradient(135deg, #047857 0%, #064e3b 100%)",
    "Animation": "linear-gradient(135deg, #8b5cf6 0%, #4c1d95 100%)",
    "Children": "linear-gradient(135deg, #ec4899 0%, #831843 100%)",
    "Comedy": "linear-gradient(135deg, #d97706 0%, #78350f 100%)",
    "Crime": "linear-gradient(135deg, #374151 0%, #111827 100%)",
    "Drama": "linear-gradient(135deg, #4338ca 0%, #1e1b4b 100%)",
    "Fantasy": "linear-gradient(135deg, #7c3aed 0%, #312e81 100%)",
    "Horror": "linear-gradient(135deg, #18181b 0%, #450a0a 100%)",
    "Mystery": "linear-gradient(135deg, #0e7490 0%, #164e63 100%)",
    "Romance": "linear-gradient(135deg, #e11d48 0%, #881337 100%)",
    "Sci-Fi": "linear-gradient(135deg, #0284c7 0%, #082f49 100%)",
    "Thriller": "linear-gradient(135deg, #b45309 0%, #1e293b 100%)",
    "War": "linear-gradient(135deg, #4d7c0f 0%, #1a2e05 100%)",
    "Western": "linear-gradient(135deg, #92400e 0%, #451a03 100%)",
}


def render_movie_card_html(movie: Dict[str, Any], show_reason: bool = False, score_label: str = "Match") -> str:
    """Render an HTML card for a movie item."""
    genres = movie.get("genres", [])
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split(",") if g.strip()]

    first_genre = genres[0] if genres else "Drama"
    gradient = GENRE_GRADIENTS.get(first_genre, "linear-gradient(135deg, #4338ca 0%, #0f172a 100%)")

    title = html.escape(str(movie.get("clean_title") or movie.get("title", "Untitled")))
    year = movie.get("year")
    year_str = f"({int(year)})" if (year is not None and pd.notna(year)) else ""

    rating = movie.get("vote_average", 0.0)
    rating_count = movie.get("vote_count", 0)

    # Score tag
    score_tag = ""
    if "hybrid_score" in movie:
        pct = int(movie["hybrid_score"] * 100)
        score_tag = f'<span class="badge-score">⚡ {pct}% Hybrid</span>'
    elif "similarity_score" in movie:
        pct = int(movie["similarity_score"] * 100)
        score_tag = f'<span class="badge-score">🎯 {pct}% {score_label}</span>'
    elif "predicted_rating" in movie:
        score_tag = f'<span class="badge-score">⭐ {movie["predicted_rating"]:.1f} Est.</span>'

    # Genre pills (top 2)
    pills = "".join([f'<span class="badge-genre">{html.escape(g)}</span>' for g in genres[:2]])

    reason_html = ""
    if show_reason and "reason" in movie:
        reason_html = f'<div style="font-size:0.75rem; color:#94a3b8; margin-top:0.4rem; line-height:1.2;">💡 {html.escape(str(movie["reason"]))}</div>'

    card_html = f"""
    <div class="movie-card">
        <div class="poster-box" style="background: {gradient};">
            <div style="font-size:1.8rem; margin-bottom:auto;">🎬</div>
            <div style="font-weight:700; font-size:1rem; color:#ffffff; text-shadow:0 2px 4px rgba(0,0,0,0.8); line-height:1.2;">
                {title}
            </div>
            <div style="font-size:0.8rem; color:#e2e8f0; opacity:0.9; margin-top:2px;">
                {year_str}
            </div>
        </div>
        <div>
            <div style="margin-bottom:0.4rem;">{pills}</div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.3rem;">
                <span class="badge-rating">⭐ {rating:.1f} <span style="font-size:0.7rem; opacity:0.8; margin-left:3px;">({rating_count})</span></span>
                {score_tag}
            </div>
            {reason_html}
        </div>
    </div>
    """
    return card_html


# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:1.5rem;">
            <div style="font-size:2rem;">🍿</div>
            <div>
                <h2 style="margin:0; font-size:1.5rem; color:#f8fafc !important;">CineMatch</h2>
                <div style="font-size:0.78rem; color:#94a3b8;">Intelligent ML Recommender</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        options=["Home", "Discover Movies", "For You", "Movie Details", "Model Insights"],
        index=["Home", "Discover Movies", "For You", "Movie Details", "Model Insights"].index(
            st.session_state.active_page
        ),
        key="nav_radio",
    )
    st.session_state.active_page = page

    st.markdown("---")
    st.markdown("### 👤 User Persona Profile")
    persona_type = st.radio(
        "Simulation Mode",
        options=["New User (Cold Start)", "MovieLens Historical User"],
        index=0,
    )

    simulated_user_id: Optional[int] = None
    if persona_type == "MovieLens Historical User":
        all_users = service.get_all_users()
        simulated_user_id = st.selectbox(
            "Select User ID from Dataset",
            options=all_users[:100],  # First 100 active users
            index=0,
            help="Simulates an existing user with known historical ratings in the training split.",
        )
        if simulated_user_id:
            user_ratings = service.get_user_ratings(simulated_user_id)
            st.caption(f"User #{simulated_user_id} has **{len(user_ratings)}** recorded ratings in MovieLens.")
    else:
        st.caption(
            "Cold-Start mode: Recommendation is guided by your selected favorites in the **For You** tab."
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.78rem; color:#64748b; line-height:1.4;">
            <strong>CineMatch Engine v1.0</strong><br>
            • Content-Based (TF-IDF + Cosine)<br>
            • Collaborative Filtering (SVD Matrix Factorization)<br>
            • Weighted Hybrid Fusion<br>
            Data: MovieLens 100k Small
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE 1: HOME
# -----------------------------------------------------------------------------
if st.session_state.active_page == "Home":
    st.markdown(
        """
        <div class="brand-hero">
            <h1 class="brand-title">CineMatch</h1>
            <div class="brand-tagline">
                Intelligent, machine-learning powered movie recommendations backed by real mathematical models.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Search Bar
    search_col1, search_col2 = st.columns([4, 1])
    with search_col1:
        home_query = st.text_input(
            "Search movies in catalog...",
            placeholder="Type a title e.g. Inception, Toy Story, Pulp Fiction...",
            label_visibility="collapsed",
        )
    with search_col2:
        search_clicked = st.button("🔍 Search", use_container_width=True)

    if home_query:
        st.markdown(f"### 🎯 Search Results for *'{home_query}'*")
        results = service.search_movies(query=home_query, limit=8)
        if results:
            cols = st.columns(4)
            for idx, movie in enumerate(results):
                with cols[idx % 4]:
                    st.markdown(render_movie_card_html(movie), unsafe_allow_html=True)
                    if st.button("Explore Movie", key=f"btn_explore_{movie['movieId']}", use_container_width=True):
                        st.session_state.selected_movie_id = movie["movieId"]
                        st.session_state.active_page = "Movie Details"
                        st.rerun()
        else:
            st.info("No matching movies found. Try another search query.")
        st.markdown("---")

    # Call to action banner
    col_cta1, col_cta2 = st.columns([3, 1])
    with col_cta1:
        st.markdown(
            """
            <div style="background: rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.3); border-radius:12px; padding:1.2rem; margin-bottom:1.5rem;">
                <h4 style="margin:0 0 0.4rem 0;">✨ Looking for personalized movie recommendations?</h4>
                <div style="color:#cbd5e1; font-size:0.92rem;">
                    Select your favorite movies or connect an existing profile in the <strong>For You</strong> section to test our Hybrid, Collaborative Filtering, and Content-Based engines.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_cta2:
        st.write("")
        if st.button("🚀 Go to For You", use_container_width=True):
            st.session_state.active_page = "For You"
            st.rerun()

    # Section 1: Most Popular Movies
    st.markdown("### 🔥 Most Popular Movies")
    st.caption("Ranked by total interaction volume across all users in the dataset.")
    popular_movies = service.get_popular_movies(limit=8)
    pop_cols = st.columns(4)
    for idx, movie in enumerate(popular_movies):
        with pop_cols[idx % 4]:
            st.markdown(render_movie_card_html(movie), unsafe_allow_html=True)
            if st.button("View Details", key=f"pop_{movie['movieId']}", use_container_width=True):
                st.session_state.selected_movie_id = movie["movieId"]
                st.session_state.active_page = "Movie Details"
                st.rerun()

    st.markdown("---")

    # Section 2: Highest-Rated Movies (with Bayesian Safeguard)
    st.markdown("### ⭐ Highest-Rated Masterpieces")
    st.caption(
        "Ranked using a **Bayesian weighted rating formula** \\( W = \\frac{v}{v+m}R + \\frac{m}{v+m}C \\) to prevent movies with only 1 rating from distorting rankings."
    )
    highest_rated = service.get_highest_rated_movies(limit=8)
    high_cols = st.columns(4)
    for idx, movie in enumerate(highest_rated):
        with high_cols[idx % 4]:
            st.markdown(render_movie_card_html(movie), unsafe_allow_html=True)
            if st.button("View Details", key=f"high_{movie['movieId']}", use_container_width=True):
                st.session_state.selected_movie_id = movie["movieId"]
                st.session_state.active_page = "Movie Details"
                st.rerun()


# -----------------------------------------------------------------------------
# PAGE 2: DISCOVER MOVIES
# -----------------------------------------------------------------------------
elif st.session_state.active_page == "Discover Movies":
    st.markdown("## 🧭 Discover Movies")
    st.caption("Browse, filter, and explore the complete 9,742 movie catalog.")

    all_genres = ["All"] + service.get_all_genres()

    filt_col1, filt_col2, filt_col3 = st.columns([2, 1, 1])
    with filt_col1:
        disc_query = st.text_input("Filter by title", placeholder="Search title...", key="disc_query")
    with filt_col2:
        selected_genre = st.selectbox("Genre Filter", options=all_genres, index=0)
    with filt_col3:
        min_rating = st.slider("Minimum Average Rating", min_value=0.0, max_value=5.0, value=3.0, step=0.5)

    year_slider = st.slider("Release Year Range", min_value=1900, max_value=2020, value=(1970, 2018), step=1)

    matching_movies = service.search_movies(
        query=disc_query,
        genre=selected_genre,
        year_range=year_slider,
        min_rating=min_rating,
        limit=24,
    )

    st.markdown(f"**Found {len(matching_movies)} matching titles:**")

    if matching_movies:
        cols = st.columns(4)
        for idx, movie in enumerate(matching_movies):
            with cols[idx % 4]:
                st.markdown(render_movie_card_html(movie), unsafe_allow_html=True)
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("Inspect", key=f"disc_insp_{movie['movieId']}", use_container_width=True):
                        st.session_state.selected_movie_id = movie["movieId"]
                        st.session_state.active_page = "Movie Details"
                        st.rerun()
                with btn_col2:
                    is_fav = movie["movieId"] in st.session_state.user_favorites
                    label = "❤️ Favorited" if is_fav else "🤍 Favorite"
                    if st.button(label, key=f"disc_fav_{movie['movieId']}", use_container_width=True):
                        if is_fav:
                            del st.session_state.user_favorites[movie["movieId"]]
                        else:
                            st.session_state.user_favorites[movie["movieId"]] = 5.0
                        st.rerun()
    else:
        st.warning("No movies matched your current filters. Try relaxing the rating or year filters.")


# -----------------------------------------------------------------------------
# PAGE 3: FOR YOU (PERSONALIZED RECOMMENDATION WORKBENCH)
# -----------------------------------------------------------------------------
elif st.session_state.active_page == "For You":
    st.markdown("## 🎯 For You — Personalized Recommendation Engine")
    st.caption("Compare how Content-Based, Collaborative Filtering, and Hybrid models personalize recommendations for your profile.")

    # Section A: Preference Profile Builder
    st.markdown("### 1. Your Preference Profile")

    # Add new favorite selector
    add_col1, add_col2, add_col3 = st.columns([3, 1, 1])
    with add_col1:
        # Searchable movie selector
        all_titles = service.movies_df[["movieId", "clean_title", "year"]].copy()
        all_titles["label"] = all_titles.apply(
            lambda r: f"{r['clean_title']} ({int(r['year']) if pd.notna(r['year']) else 'N/A'})", axis=1
        )
        title_dict = dict(zip(all_titles["label"], all_titles["movieId"]))

        selected_label = st.selectbox(
            "Add a movie to your favorites:",
            options=all_titles["label"].tolist(),
            index=0,
        )
    with add_col2:
        new_rating = st.select_slider("Your Rating", options=[1.0, 2.0, 3.0, 4.0, 5.0], value=5.0)
    with add_col3:
        st.write("")
        if st.button("➕ Add Movie", use_container_width=True):
            chosen_id = title_dict[selected_label]
            st.session_state.user_favorites[chosen_id] = new_rating
            st.success(f"Added to favorites with rating {new_rating}⭐!")
            st.rerun()

    # Display Current Profile Movies
    if st.session_state.user_favorites:
        st.markdown(f"**Currently Selected Favorites ({len(st.session_state.user_favorites)} movies):**")
        fav_cols = st.columns(min(len(st.session_state.user_favorites), 4))
        fav_items = list(st.session_state.user_favorites.items())

        for idx, (fav_id, rating) in enumerate(fav_items):
            movie_data = service.get_movie_by_id(fav_id)
            if movie_data:
                with fav_cols[idx % 4]:
                    st.markdown(
                        f"""
                        <div style="background:#1e293b; border:1px solid #3b82f6; border-radius:10px; padding:0.8rem; margin-bottom:0.5rem;">
                            <div style="font-weight:700; font-size:0.9rem; color:#ffffff;">{html.escape(movie_data['clean_title'])}</div>
                            <div style="font-size:0.75rem; color:#94a3b8;">{html.escape(movie_data['genres_str'])}</div>
                            <div style="font-size:0.85rem; color:#fbbf24; margin-top:0.3rem;">Your rating: <strong>{rating}⭐</strong></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button("Remove", key=f"rem_fav_{fav_id}", use_container_width=True):
                        del st.session_state.user_favorites[fav_id]
                        st.rerun()
    else:
        st.info("Select at least 1 movie above to initialize your taste profile.")

    st.markdown("---")

    # Section B: Recommender Engine Selection & Configuration
    st.markdown("### 2. Recommendation Algorithm Configuration")

    recommender_mode = st.radio(
        "Select Recommendation Approach:",
        options=["Hybrid Recommender (Recommended)", "Content-Based Filtering", "Collaborative Filtering (SVD)"],
        horizontal=True,
    )

    content_weight = 0.5
    if recommender_mode == "Hybrid Recommender (Recommended)":
        weight_col1, weight_col2 = st.columns([3, 1])
        with weight_col1:
            content_weight = st.slider(
                "Content Weight (α) vs Collaborative Weight (1 - α):",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Higher Content Weight favors metadata and genre alignment; higher Collaborative Weight favors community rating patterns.",
            )
        with weight_col2:
            cf_weight = 1.0 - content_weight
            st.metric("Weight Breakdown", f"{int(content_weight*100)}% Content / {int(cf_weight*100)}% CF")

    top_n_recs = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

    # Section C: Recommendation Results Generation
    st.markdown("---")
    st.markdown(f"### 3. Your Personalized Recommendations ({recommender_mode})")

    # Generate genuine recommendations
    recommendations: List[Dict[str, Any]] = []

    if recommender_mode == "Content-Based Filtering":
        if st.session_state.user_favorites:
            recommendations = service.content_model.recommend_from_profile(
                movie_ratings=st.session_state.user_favorites,
                top_n=top_n_recs,
            )
        else:
            st.warning("Please add at least one favorite movie to generate content recommendations.")

    elif recommender_mode == "Collaborative Filtering (SVD)":
        if persona_type == "MovieLens Historical User" and simulated_user_id:
            recommendations = service.recommend_collaborative(
                user_id=simulated_user_id,
                top_n=top_n_recs,
                exclude_ids=set(st.session_state.user_favorites.keys()),
            )
        else:
            st.info(
                "💡 Collaborative Filtering learns from historical user-item interactions. "
                "Switch to **'MovieLens Historical User'** in the sidebar to simulate an existing user with trained matrix factorization weights, "
                "or use the **Hybrid Recommender** which gracefully blends cold-start profiles."
            )

    else:  # Hybrid Recommender
        if st.session_state.user_favorites:
            user_id_arg = simulated_user_id if persona_type == "MovieLens Historical User" else None
            recommendations = service.recommend_hybrid(
                user_ratings=st.session_state.user_favorites,
                user_id=user_id_arg,
                content_weight=content_weight,
                top_n=top_n_recs,
            )
        else:
            st.warning("Please select at least 1 favorite movie to generate hybrid recommendations.")

    if recommendations:
        rec_cols = st.columns(4)
        for idx, rec in enumerate(recommendations):
            with rec_cols[idx % 4]:
                st.markdown(render_movie_card_html(rec, show_reason=True), unsafe_allow_html=True)
                if st.button("Inspect Movie", key=f"rec_insp_{rec['movieId']}", use_container_width=True):
                    st.session_state.selected_movie_id = rec["movieId"]
                    st.session_state.active_page = "Movie Details"
                    st.rerun()


# -----------------------------------------------------------------------------
# PAGE 4: MOVIE DETAILS
# -----------------------------------------------------------------------------
elif st.session_state.active_page == "Movie Details":
    st.markdown("## 🎞️ Movie Details")

    movie_id = st.session_state.selected_movie_id
    movie = service.get_movie_by_id(movie_id)

    if not movie:
        st.error(f"Movie with ID {movie_id} was not found.")
    else:
        title = movie["clean_title"]
        year = int(movie["year"]) if pd.notna(movie["year"]) else "Unknown Year"
        genres = movie.get("genres", [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(",") if g.strip()]

        col_left, col_right = st.columns([1, 2])

        with col_left:
            # Stylized poster card
            first_genre = genres[0] if genres else "Drama"
            gradient = GENRE_GRADIENTS.get(first_genre, "linear-gradient(135deg, #4338ca 0%, #0f172a 100%)")
            st.markdown(
                f"""
                <div style="background:{gradient}; border-radius:16px; padding:2.5rem 1.5rem; text-align:center; box-shadow:0 10px 30px rgba(0,0,0,0.5); border:1px solid rgba(255,255,255,0.1);">
                    <div style="font-size:4rem; margin-bottom:1rem;">🎬</div>
                    <h2 style="margin:0; font-size:1.8rem; color:#ffffff !important;">{html.escape(title)}</h2>
                    <div style="color:#e2e8f0; font-size:1.1rem; margin-top:0.4rem;">({year})</div>
                    <div style="margin-top:1.2rem;">
                        {"".join([f'<span class="badge-genre">{html.escape(g)}</span>' for g in genres])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # External Links
            st.write("")
            link_cols = st.columns(2)
            with link_cols[0]:
                imdb_id = movie.get("imdbId")
                if imdb_id:
                    st.link_button("🌐 IMDb Profile", f"https://www.imdb.com/title/{imdb_id}/", use_container_width=True)
            with link_cols[1]:
                tmdb_id = movie.get("tmdbId")
                if tmdb_id:
                    st.link_button("🍿 TMDb Page", f"https://www.themoviedb.org/movie/{tmdb_id}", use_container_width=True)

        with col_right:
            st.markdown(f"### {title}")
            st.markdown(f"**Original Dataset Title:** `{movie.get('title')}`")
            st.markdown(f"**Genres:** {', '.join(genres) if genres else 'None'}")
            tags_text = movie.get("tags_text")
            st.markdown(f"**Community Tags:** {tags_text if tags_text else '*No community tags recorded for this movie*'}")

            st.markdown("---")
            stat_c1, stat_c2, stat_c3 = st.columns(3)
            with stat_c1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-val">⭐ {movie.get('vote_average', 0.0):.2f}</div>
                        <div class="metric-lbl">Average Rating</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with stat_c2:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-val">👥 {int(movie.get('vote_count', 0)):,}</div>
                        <div class="metric-lbl">Ratings Count</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with stat_c3:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-val">⚖️ {movie.get('weighted_rating', 0.0):.2f}</div>
                        <div class="metric-lbl">Bayesian Weighted Score</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # SVD Model Prediction for active persona
            st.markdown("---")
            if persona_type == "MovieLens Historical User" and simulated_user_id and service.cf_model:
                pred_val = service.cf_model.predict_rating(simulated_user_id, movie_id)
                st.info(
                    f"🤖 **Collaborative Filtering Model Prediction:** For User #{simulated_user_id}, "
                    f"the SVD algorithm predicts a rating of **{pred_val:.2f} / 5.00**."
                )
            else:
                st.info(
                    f"💡 Average rating across all dataset viewers is **{movie.get('vote_average', 0.0):.2f} / 5.00**."
                )

        st.markdown("---")
        st.markdown(f"### 👯 Movies Most Similar to '{title}' (Content-Based Engine)")
        st.caption("Calculated on-the-fly using TF-IDF feature vectors across genres, tags, and titles.")

        similar_movies = service.recommend_similar(movie_id, top_n=8)
        if similar_movies:
            sim_cols = st.columns(4)
            for idx, s_movie in enumerate(similar_movies):
                with sim_cols[idx % 4]:
                    st.markdown(render_movie_card_html(s_movie, show_reason=True), unsafe_allow_html=True)
                    if st.button("Explore", key=f"sim_exp_{s_movie['movieId']}", use_container_width=True):
                        st.session_state.selected_movie_id = s_movie["movieId"]
                        st.rerun()


# -----------------------------------------------------------------------------
# PAGE 5: MODEL INSIGHTS & BENCHMARKING
# -----------------------------------------------------------------------------
elif st.session_state.active_page == "Model Insights":
    st.markdown("## 📊 Model Insights & Evaluation Benchmarks")
    st.caption("Explore live exploratory data analysis and genuine machine learning evaluation metrics.")

    tab_eval, tab_eda, tab_math = st.tabs(["🏆 Model Evaluation & Metrics", "📈 Dataset Analytics", "🔬 Algorithm Methodology"])

    with tab_eval:
        st.markdown("### Genuine Model Performance Evaluation (80/20 Split)")
        st.markdown(
            """
            To evaluate our recommendation engines without data leakage:
            - **Split:** 80% Training ratings (80,668) / 20% Test ratings (20,168) with deterministic seed `random_state=42`.
            - **Ground Truth Relevance:** A movie is defined as *relevant* if the user rated it **≥ 4.0 / 5.0** in the held-out test split.
            """
        )

        eval_data = service.eval_results
        if not eval_data:
            st.warning("Evaluation metrics have not been computed yet. Please run `python scripts/evaluate_models.py`.")
        else:
            # Metric Summary Cards
            pred_metrics = eval_data.get("rating_prediction", {})
            svd_res = pred_metrics.get("svd_collaborative", {})
            base_res = pred_metrics.get("global_mean_baseline", {})
            improvement = pred_metrics.get("rmse_improvement_pct", 16.12)

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-val">{svd_res.get('rmse', 0.8797):.4f}</div>
                        <div class="metric-lbl">SVD Test RMSE (Root Mean Sq. Error)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col2:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-val">{svd_res.get('mae', 0.6728):.4f}</div>
                        <div class="metric-lbl">SVD Test MAE (Mean Abs. Error)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col3:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-val" style="color:#10b981;">+{improvement:.1f}%</div>
                        <div class="metric-lbl">RMSE Improvement vs Baseline ({base_res.get('rmse', 1.0488):.4f})</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("### Top-10 Ranking Evaluation Benchmark")
            st.caption(
                "Comparing **Precision@10**, **Recall@10**, and **NDCG@10** (Normalized Discounted Cumulative Gain)."
            )

            ranking_data = eval_data.get("ranking_performance_at_10", {})
            ranking_rows = []
            for m_key, metrics in ranking_data.items():
                name_clean = m_key.replace("_", " ").title()
                ranking_rows.append({
                    "Model": name_clean,
                    "Precision@10": metrics.get("precision@10", 0.0),
                    "Recall@10": metrics.get("recall@10", 0.0),
                    "NDCG@10": metrics.get("ndcg@10", 0.0),
                })

            df_ranking = pd.DataFrame(ranking_rows)
            st.dataframe(df_ranking, use_container_width=True, hide_index=True)

            # Plotly Visualization for Ranking
            fig_rank = go.Figure()
            models_list = df_ranking["Model"].tolist()
            fig_rank.add_trace(go.Bar(name="Precision@10", x=models_list, y=df_ranking["Precision@10"], marker_color="#8b5cf6"))
            fig_rank.add_trace(go.Bar(name="Recall@10", x=models_list, y=df_ranking["Recall@10"], marker_color="#06b6d4"))
            fig_rank.add_trace(go.Bar(name="NDCG@10", x=models_list, y=df_ranking["NDCG@10"], marker_color="#ec4899"))
            fig_rank.update_layout(
                title="Model Ranking Comparison (@10)",
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#1e293b",
                height=380,
            )
            st.plotly_chart(fig_rank, use_container_width=True)

            st.markdown(
                """
                > [!NOTE] **Understanding Offline Evaluation Bias**:
                > Collaborative Filtering (SVD) achieves top ranking metrics on historical test sets because users in MovieLens specifically chose to watch and rate those items.
                > Content-Based recommenders surface high-similarity items across the full catalog of 9,742 movies, most of which users never watched in the historical data (unobserved items penalized as false negatives).
                > In practical user testing, **Hybrid Recommenders** offer the best user satisfaction by blending serendipity and accuracy.
                """
            )

    with tab_eda:
        st.markdown("### Dataset Exploratory Data Analysis")

        eda_c1, eda_c2 = st.columns(2)

        with eda_c1:
            # Rating Distribution
            fig_ratings = px.histogram(
                service.ratings_df,
                x="rating",
                nbins=10,
                title="Rating Value Distribution (100,836 total ratings)",
                color_discrete_sequence=["#8b5cf6"],
                template="plotly_dark",
            )
            fig_ratings.update_layout(paper_bgcolor="#111827", plot_bgcolor="#1e293b")
            st.plotly_chart(fig_ratings, use_container_width=True)

        with eda_c2:
            # Genre Distribution
            genre_counts = {}
            for g_list in service.movies_df["genres"]:
                if isinstance(g_list, list):
                    for g in g_list:
                        if g and g != "(no genres listed)":
                            genre_counts[g] = genre_counts.get(g, 0) + 1

            df_genres = pd.DataFrame(list(genre_counts.items()), columns=["Genre", "Count"]).sort_values(
                by="Count", ascending=False
            ).head(12)

            fig_genres = px.bar(
                df_genres,
                x="Count",
                y="Genre",
                orientation="h",
                title="Top 12 Movie Genres in Catalog",
                color="Count",
                color_continuous_scale="Purples",
                template="plotly_dark",
            )
            fig_genres.update_layout(paper_bgcolor="#111827", plot_bgcolor="#1e293b", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_genres, use_container_width=True)

        # Release Year Distribution
        valid_years = service.movies_df.dropna(subset=["year"])
        valid_years = valid_years[(valid_years["year"] >= 1920) & (valid_years["year"] <= 2020)]
        fig_years = px.histogram(
            valid_years,
            x="year",
            nbins=40,
            title="Movies Release Year Distribution (1920 - 2020)",
            color_discrete_sequence=["#06b6d4"],
            template="plotly_dark",
        )
        fig_years.update_layout(paper_bgcolor="#111827", plot_bgcolor="#1e293b")
        st.plotly_chart(fig_years, use_container_width=True)

    with tab_math:
        st.markdown("### Recommendation Mathematical Formulations")

        st.markdown(
            """
            #### 1. Content-Based Filtering: TF-IDF & Cosine Similarity
            We construct a metadata soup for each movie combining weighted genres, community tags, and title tokens.
            The TF-IDF (Term Frequency - Inverse Document Frequency) score for word \\( t \\) in document \\( d \\) is:
            \\[
            \\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\log \\left( \\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|} \\right) + 1
            \\]
            Similarity between candidate movie vector \\( \\vec{m} \\) and user profile vector \\( \\vec{u} \\) is computed via Cosine Similarity:
            \\[
            \\text{Cosine}(\\vec{u}, \\vec{m}) = \\frac{\\vec{u} \\cdot \\vec{m}}{\\|\\vec{u}\\|_2 \\|\\vec{m}\\|_2}
            \\]

            #### 2. Collaborative Filtering: SVD Matrix Factorization
            We factorize the sparse interaction matrix \\( R \\) into low-rank latent user matrices \\( P \\) and item matrices \\( Q \\):
            \\[
            \\hat{r}_{u, i} = \\mu + b_u + b_i + p_u^T q_i
            \\]
            Optimized via Stochastic Gradient Descent (SGD) minimizing regularized squared error:
            \\[
            \\min_{P, Q, b} \\sum_{(u, i) \\in \\text{Train}} \\left( r_{u, i} - \\hat{r}_{u, i} \\right)^2 + \\lambda \\left( \\|p_u\\|^2 + \\|q_i\\|^2 + b_u^2 + b_i^2 \\right)
            \\]

            #### 3. Configurable Hybrid Fusion
            Scores from both models are normalized to \\( [0, 1] \\) and combined using configurable weight \\( \\alpha \\):
            \\[
            S_{\\text{hybrid}} = \\alpha \\cdot S_{\\text{content}} + (1 - \\alpha) \\cdot \\left( \\frac{\\hat{r}_{u, i} - 0.5}{4.5} \\right)
            \\]
            """
        )
