# modules/umap_view.py
# -------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import scipy.cluster.hierarchy as sch

# Safe import of umap-learn
try:
    from umap import UMAP
except ModuleNotFoundError:
    st.error("`umap-learn` is not installed. Run: pip install umap-learn  (or conda install -c conda-forge umap-learn)")
    st.stop()

# ---------------- Look-back map ----------------
_LOOKBACK_DAYS = {"6M": 126, "1Y": 252, "3Y": 756}

# ---------------- Sidebar controls ----------------
def _sidebar_controls():
    st.sidebar.markdown("**UMAP Controls**")
    book   = st.sidebar.selectbox("Book", ["All", "Long", "Short"], index=0)
    window = st.sidebar.selectbox("Look-back", list(_LOOKBACK_DAYS.keys()), index=1)
    n_neighbors = st.sidebar.slider("n_neighbors", 5, 50, 15, 1,
                                    help="Larger = more global structure, smaller = more local")
    min_dist    = st.sidebar.slider("min_dist", 0.0, 0.99, 0.1, 0.01,
                                    help="Smaller keeps clusters tighter")
    metric      = st.sidebar.selectbox("Distance metric", ["euclidean","manhattan","cosine"], index=0)
    show_table  = st.sidebar.checkbox("Show coordinates table", value=False)
    return book, window, n_neighbors, min_dist, metric, show_table

# ---------------- Preprocess returns ----------------
def _slice_returns(full_df: pd.DataFrame,
                   book: str,
                   window: str,
                   longs: pd.DataFrame,
                   shorts: pd.DataFrame) -> pd.DataFrame:
    df = full_df.copy()
    if "date" in df.columns:
        df = df.drop(columns="date")
    df.index = pd.to_datetime(df.index)
    df = df.iloc[-_LOOKBACK_DAYS[window]:]

    if book == "Long":
        df = df.reindex(columns=longs.columns).dropna(axis=1, how="all")
    elif book == "Short":
        df = df.reindex(columns=shorts.columns).dropna(axis=1, how="all")

    # drop assets with >25% missing, ffill then bfill limit=5
    df = df.dropna(axis=1, thresh=int(0.75 * len(df)))
    df = df.ffill().bfill(limit=5)
    return df

# ---------------- Optional: reuse cluster labels (not required) ----------------
def _clusters_for_color(Y: pd.DataFrame):
    """Quick Ward clustering to optionally color by cluster if needed."""
    codep = Y.corr(method="spearman")
    dist  = np.sqrt(0.5 * (1 - codep))
    cond  = sch.distance.squareform(dist.values)
    link  = sch.linkage(cond, method="ward", optimal_ordering=True)

    # simple cut: max 8 clusters or number of assets//3
    max_k = min(8, max(2, Y.shape[1] // 3))
    labels = sch.fcluster(link, t=max_k, criterion="maxclust")
    return pd.Series(labels, index=Y.columns, name="Cluster")

# ---------------- Cached embed ----------------
@st.cache_data(show_spinner="Embedding with UMAP...")
def _umap_embed(Y: pd.DataFrame,
                n_neighbors: int,
                min_dist: float,
                metric: str) -> pd.DataFrame:
    """
    Y: returns (rows = time, cols = tickers)
    Returns: DataFrame with x, y for each ticker
    """
    # X shape = n_assets x n_time
    X = Y.T.values
    reducer = UMAP(n_neighbors=n_neighbors,
                   min_dist=min_dist,
                   metric=metric,
                   random_state=42)
    coords = reducer.fit_transform(X)
    out = pd.DataFrame(coords, columns=["UMAP1","UMAP2"], index=Y.columns)
    return out

# ---------------- Public render ----------------
def render_umap(master_df, longs_df, shorts_df, portfolio_df, page_header):
    page_header("🌀", "UMAP Projection", "2D embedding of assets by return patterns")

    book, window, n_neighbors, min_dist, metric, show_table = _sidebar_controls()

    # 1. Slice data
    Y = _slice_returns(master_df, book, window, longs_df, shorts_df)
    if Y.empty:
        st.warning("No return data after slicing/cleaning.")
        return

    # 2. Get meta for coloring/size
    port = portfolio_df.set_index("EOD Ticker", drop=False)
    # Sector
    sectors = port.reindex(Y.columns)["Sector"].fillna("Unknown")
    # |net weight| for bubble size
    if "Net_weight" in port.columns:
        bubble = port.reindex(Y.columns)["Net_weight"].abs().fillna(0)
    else:
        bubble = pd.Series(1.0, index=Y.columns)  # fallback

    # 3. Compute UMAP coords
    coords = _umap_embed(Y, n_neighbors, min_dist, metric)

    plot_df = coords.copy()
    plot_df["Ticker"] = plot_df.index
    plot_df["Sector"] = sectors.values
    plot_df["|Net_w|"] = bubble.values

    # 4. Make scatter
    fig = px.scatter(
        plot_df,
        x="UMAP1", y="UMAP2",
        color="Sector",
        size="|Net_w|",
        hover_name="Ticker",
        template="plotly_dark",
        title=f"UMAP ({book}, {window}) – n_neighbors={n_neighbors}, min_dist={min_dist}"
    )
    fig.update_layout(
        legend_title_text="Sector",
        margin=dict(l=10, r=10, t=60, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 5. Optional table
    if show_table:
        with st.expander("UMAP Coordinates Table"):
            st.dataframe(plot_df[["Ticker","Sector","UMAP1","UMAP2","|Net_w|"]]
                         .sort_values("Sector"), use_container_width=True)