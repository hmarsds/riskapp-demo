# modules/umap.py
# -------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import scipy.cluster.hierarchy as sch

# import umap lazily so the whole app still runs if it's missing
try:
    import umap
except ImportError:
    umap = None

# ---------------- Look‑back map ----------------
_LOOKBACK_DAYS = {"6M": 126, "1Y": 252, "3Y": 756}

# -------------------- Helpers --------------------
def _slice_returns(full_df: pd.DataFrame, book: str, window: str,
                   longs: pd.DataFrame, shorts: pd.DataFrame) -> pd.DataFrame:
    df = full_df.copy()
    if "date" in df.columns:
        df = df.drop(columns="date")
    df.index = pd.to_datetime(df.index)
    df = df.iloc[-_LOOKBACK_DAYS[window]:]

    if book == "Long":
        df = df.reindex(columns=longs.columns).dropna(axis=1, how="all")
    elif book == "Short":
        df = df.reindex(columns=shorts.columns).dropna(axis=1, how="all")

    df = df.dropna(axis=1, thresh=int(0.75 * len(df)))
    df = df.ffill().bfill(limit=5)
    return df

def _compute_clusters(Y: pd.DataFrame, max_k: int = 10, min_size: int = 2):
    """Ward linkage on Spearman distance + auto-k (two-diff gap)."""
    codep = Y.corr(method="spearman")
    dist  = np.sqrt(0.5 * (1 - codep))
    cond  = sch.distance.squareform(dist.values)
    link  = sch.linkage(cond, method="ward", optimal_ordering=True)

    heights = np.sort(link[:, 2])[-(max_k + 1):]
    diffs   = np.diff(heights)
    ddiffs  = diffs[1:] - diffs[:-1]
    best    = np.argmax(ddiffs) + 1
    n_leaves = link.shape[0] + 1
    k0       = n_leaves - (best + 1)
    for kk in range(k0, 1, -1):
        labs = sch.fcluster(link, t=kk, criterion="maxclust")
        if np.bincount(labs)[1:].min() >= min_size:
            k0 = kk
            break
    labels = sch.fcluster(link, t=k0, criterion="maxclust")
    return labels

@st.cache_data(show_spinner=False)
def _embed_umap(returns: pd.DataFrame,
                n_neighbors: int,
                min_dist: float,
                metric: str,
                seed: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed
    )
    # assets as samples → transpose
    return reducer.fit_transform(returns.T)

def _prep_meta(portfolio_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    meta_cols = ["EOD Ticker", "Sector", "Category", "Net_weight"]
    if "EOD Ticker" not in portfolio_df.columns:
        return pd.DataFrame(index=tickers)
    meta = (portfolio_df[portfolio_df["EOD Ticker"].isin(tickers)]
            .set_index("EOD Ticker")
            .reindex(columns=[c for c in meta_cols if c in portfolio_df.columns]))
    return meta.reindex(tickers)

def _download_csv(df: pd.DataFrame, name: str):
    return df.to_csv(index=False).encode()

# -------------------- Public entry --------------------
def render_umap(master_df, longs_df, shorts_df, portfolio_df, page_header):
    page_header("🌀", "UMAP Projection",
                "2‑D embedding of assets (similar return profiles lie close)")

    if umap is None:
        st.error("`umap-learn` is not installed. `pip install umap-learn`.")
        return

    # -------- Sidebar controls --------
    with st.sidebar:
        st.markdown("**UMAP Controls**")
        book    = st.selectbox("Book", ["All", "Long", "Short"], index=0)
        windows = st.multiselect("Look‑back windows", list(_LOOKBACK_DAYS.keys()),
                                 default=["1Y"], max_selections=3)

        seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
        n_neighbors = st.slider("n_neighbors", 5, 50, 15, 1)
        min_dist    = st.slider("min_dist", 0.0, 0.8, 0.15, 0.01)
        metric      = st.selectbox("Metric", ["euclidean", "manhattan", "cosine", "correlation"], 0)

        color_by = st.selectbox("Color points by", ["Cluster", "Sector", "Gross Weight"],
                                help="Cluster from Ward-Spearman, or portfolio metadata")
        size_by  = st.selectbox("Size points by", ["None", "Gross Weight", "|Net Weight|"],
                                index=1)
        show_labels = st.checkbox("Hover labels", value=True)

    if not windows:
        st.info("Select at least one window.")
        return

    # -------- Loop through each selected window --------
    for win in windows:
        Y = _slice_returns(master_df, book, win, longs_df, shorts_df)
        if Y.empty:
            st.warning(f"No data for {win} window.")
            continue

        labels = _compute_clusters(Y, max_k=min(10, Y.shape[1]-1))
        tickers = Y.columns.tolist()

        meta = _prep_meta(portfolio_df, tickers)
        gross_w = meta["Net_weight"].abs() if "Net_weight" in meta.columns else pd.Series(1.0, index=tickers)
        gross_w = gross_w.fillna(0.0)

        sizes = pd.Series(10.0, index=tickers)
        if size_by == "Gross Weight":
            sizes = gross_w * 200
        elif size_by == "|Net Weight|" and "Net_weight" in meta.columns:
            sizes = meta["Net_weight"].abs().fillna(0.0) * 200

        emb = _embed_umap(Y, n_neighbors, min_dist, metric, seed)

        proj_df = pd.DataFrame({
            "x": emb[:, 0],
            "y": emb[:, 1],
            "Ticker": tickers,
            "Cluster": labels
        }).set_index("Ticker")

        if "Sector" in meta.columns:
            proj_df["Sector"] = meta["Sector"]
        proj_df["GrossWeight"] = gross_w
        if "Net_weight" in meta.columns:
            proj_df["Net_weight"] = meta["Net_weight"]
        proj_df["PointSize"] = sizes

        # choose color arg
        if color_by == "Cluster":
            color_arg = "Cluster"
        elif color_by == "Sector" and "Sector" in proj_df.columns:
            color_arg = "Sector"
        else:
            color_arg = "GrossWeight"

        fig = px.scatter(
            proj_df.reset_index(),
            x="x", y="y",
            color=color_arg,
            size="PointSize",
            hover_name="Ticker" if show_labels else None,
            template="plotly_dark",
            title=f"UMAP ({win}, {book})"
        )
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.update_layout(margin=dict(l=40, r=20, t=60, b=40),
                          legend_title_text=color_by)

        with st.expander(f"UMAP Projection – {win}", expanded=True if len(windows)==1 else False):
            st.plotly_chart(fig, use_container_width=True)

            csv = _download_csv(proj_df.reset_index()[["Ticker","x","y","Cluster","Sector",
                                                       "GrossWeight","Net_weight" if "Net_weight" in proj_df.columns else "GrossWeight"]],  # dummy list to avoid error
                                f"umap_{win}_{book}.csv")
            st.download_button(f"Download embedding CSV ({win})",
                               data=csv,
                               file_name=f"umap_{win}_{book}.csv",
                               mime="text/csv")

            with st.expander("Underlying DataFrame"):
                st.dataframe(proj_df.reset_index(), use_container_width=True)

        st.markdown("---")