import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

st.set_page_config(
    page_title="Interior Material Recommender – Ethiopia",
    page_icon="🏠",
    layout="wide"
)

# Load model & data
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    DATA_DIR = Path("data")
    
    # Look for the exact file
    parquet_path = DATA_DIR / "interior_materials_enriched_20260219_1714.parquet"
    csv_path = DATA_DIR / "interior_materials_enriched_20260219_1714.csv"
    
    df = None
    embeddings = None
    
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            st.success(f"Loaded Parquet: {parquet_path.name}")
        except Exception as e:
            st.error(f"Parquet load failed: {e}")
    
    elif csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            st.success(f"Loaded CSV: {csv_path.name}")
        except Exception as e:
            st.error(f"CSV load failed: {e}")
    
    else:
        st.error("File not found:")
        st.info(f"Expected: {parquet_path} or {csv_path}")
        st.info("Files in data/: " + ", ".join([f.name for f in DATA_DIR.glob("*")]))
        st.stop()
    
    # Now safely process embeddings
    if df is not None:
        if "embedding" not in df.columns:
            st.error("Data file missing 'embedding' column – run Phase 2 again.")
            st.stop()
        
        # Convert embedding column if it's string/list representation
        if isinstance(df["embedding"].iloc[0], str):
            df["embedding"] = df["embedding"].apply(eval)
        
        embeddings = np.stack(df["embedding"].values)
    else:
        st.stop()
    
    return model, df, embeddings

# Load once
model, df, embeddings = load_resources()

# Recommender function
def recommend_materials(query_text, top_k=8, min_similarity=0.45):
    if not query_text.strip():
        return pd.DataFrame()

    query_emb = model.encode([query_text], normalize_embeddings=True)[0]
    similarities = cosine_similarity([query_emb], embeddings)[0]

    result_df = df.copy()
    result_df["similarity"] = similarities.round(3)

    result_df = result_df[result_df["similarity"] >= min_similarity]
    result_df = result_df.sort_values("similarity", ascending=False).head(top_k)

    cols = ["material_name", "category", "price_etb", "unit_norm", "similarity"]
    if "detail_url" in result_df.columns:
        cols.append("detail_url")

    return result_df[cols]

# Dashboard UI
st.title(" AI Interior Material Recommender – Ethiopia")

st.markdown("Describe your needs (style, color, budget, room, material type)")

query = st.text_area(
    "What are you looking for?",
    placeholder="Examples:\n"
                "- frosted glass partition\n"
                "- galvanized metal sheet for ceiling\n"
                "- modern blue tiles under 2000 ETB\n"
                "- white marble kitchen countertop",
    height=140
)

col1, col2 = st.columns(2)
top_k = col1.slider("Number of recommendations", 3, 15, 8)
min_sim = col2.slider("Minimum similarity score", 0.3, 0.9, 0.45, 0.05)

if st.button("Find Matches", type="primary") and query.strip():
    with st.spinner("Searching..."):
        results = recommend_materials(query, top_k=top_k, min_similarity=min_sim)

    if results.empty:
        st.warning("No matches found. Try a broader description or lower the score.")
    else:
        st.success(f"Found **{len(results)}** matching materials")
        st.dataframe(
            results.style.format({
                "price_etb": "{:,.0f} ETB",
                "similarity": "{:.3f}"
            }).background_gradient(subset=["similarity"], cmap="YlGn"),
            use_container_width=True,
            hide_index=True
        )

st.markdown("---")
st.caption("Developed by **Aklilu Abera**")
st.caption("Data from 2merkato.com ")
