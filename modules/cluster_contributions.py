# modules/cluster_contributions.py
# -------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import scipy.cluster.hierarchy as sch
import riskfolio as rf
from cvxpy import installed_solvers

# ---------------- Look‑back map ----------------
_LOOKBACK_DAYS = {"6M": 126, "1Y": 252, "3Y": 756}

# ---------------- Sidebar ----------------
def _sidebar_controls():
    st.sidebar.markdown("**Cluster Contributions Controls**")
    book   = st.sidebar.selectbox("Book", ["All", "Long", "Short"], index=0)
    window = st.sidebar.selectbox("Look‑back", list(_LOOKBACK_DAYS.keys()), index=1)
    rm     = st.sidebar.selectbox("Risk Measure", ["CVaR", "CDaR"], 0)
    alpha  = st.sidebar.slider("Tail α (CVaR/CDaR)", 0.01, 0.10, 0.05, 0.01)
    return book, window, rm, alpha

# ---------------- Preprocess returns ----------------
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

    # drop cols with >25% NaN
    df = df.dropna(axis=1, thresh=int(0.75 * len(df)))
    # ffill/bfill small gaps
    df = df.ffill().bfill(limit=5)
    return df

# ---------------- Clustering ----------------
def _compute_clusters(Y: pd.DataFrame, max_k: int = 10, min_size: int = 2):
    codep = Y.corr(method="spearman")
    dist  = np.sqrt(0.5 * (1 - codep))
    cond  = sch.distance.squareform(dist.values)
    link  = sch.linkage(cond, method="ward", optimal_ordering=True)

    # auto-k via two-diff gap, enforce min_size
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
    return labels, link, codep

# ---------------- RC with gross weights only ----------------
def _risk_contributions_gross(Y: pd.DataFrame,
                              labels: np.ndarray,
                              net_weights: pd.Series,
                              rm: str,
                              alpha: float):
    # gross |w| normalisation
    w_net   = net_weights.reindex(Y.columns).fillna(0.0)
    w_gross = w_net.abs()
    denom   = w_gross.sum() or 1.0
    w_use   = w_gross / denom

    cov = Y.cov()
    sols = installed_solvers()
    solver = next((s for s in ["CLARABEL", "ECOS", "OSQP", "SCS"] if s in sols), sols[0])

    rc_raw = rf.Risk_Contribution(
        w=w_use.to_frame("w"),
        returns=Y,
        cov=cov,
        rm=rm,
        alpha=alpha,
        solver=solver
    ).flatten()

    total_rc = rc_raw.sum() or 1.0
    df_pa = pd.DataFrame({
        "Ticker"  : Y.columns,
        "Cluster" : labels,
        "RC_raw"  : rc_raw,
        "RC_pct"  : rc_raw / total_rc * 100,
        "w_gross" : w_gross.values
    })

    tot_gross = df_pa["w_gross"].sum() or 1.0

    df_cluster = (df_pa.groupby("Cluster")
                        .agg(GrossWeight=("w_gross", "sum"),
                             Cluster_Risk_pct=("RC_pct", "sum"))
                        .assign(**{
                            "%Total Weight (gross)": lambda d: d["GrossWeight"] / tot_gross * 100
                        })
                        .reset_index())

    members = (df_pa.sort_values(["Cluster", "w_gross"], ascending=[True, False])
                    .groupby("Cluster")["Ticker"]
                    .apply(lambda s: ", ".join(s.head(5)))
                    .reset_index(name="Top Members"))

    df_cluster = df_cluster.merge(members, on="Cluster")
    return df_cluster, df_pa

# ---------------- Public render ----------------
def render_cluster_contributions(master_df, longs_df, shorts_df, portfolio_df, page_header):
    page_header("", "Cluster Contributions", "Aggregated risk by cluster (gross weights)")

    book, window, rm, alpha = _sidebar_controls()

    # 1. returns slice
    Y = _slice_returns(master_df, book, window, longs_df, shorts_df)
    if Y.empty:
        st.warning("No return data after slicing / cleaning.")
        return

    # 2. weights
    if {"EOD Ticker", "Net_weight"} - set(portfolio_df.columns):
        st.error("portfolio_df must include 'EOD Ticker' and 'Net_weight'.")
        return
    net_weights = portfolio_df.set_index("EOD Ticker")["Net_weight"]

    # 3. clusters
    labels, _, _ = _compute_clusters(Y, max_k=min(10, Y.shape[1] - 1))

    # 4. RC
    df_cluster, df_pa = _risk_contributions_gross(Y, labels, net_weights, rm, alpha)

    # 5. sort for chart (largest RC left)
    df_sorted = df_cluster.sort_values("Cluster_Risk_pct", ascending=False).reset_index(drop=True)
    df_sorted["Cluster_str"] = df_sorted["Cluster"].astype(str)
    order = df_sorted["Cluster_str"].tolist()

    fig = px.bar(
        df_sorted,
        x="Cluster_str",                 # string axis
        y="Cluster_Risk_pct",
        template="plotly_dark",
        labels={"Cluster_Risk_pct": "% Risk Contribution", "Cluster_str": "Cluster"},
        title=f"{rm} Cluster Risk Contributions ({window}, {book})"
    )
    fig.update_layout(
        xaxis=dict(categoryorder="array", categoryarray=order),
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 6. summary table
    table_cols = ["Cluster", "Top Members", "%Total Weight (gross)", "Cluster_Risk_pct"]
    st.dataframe(
        df_sorted[table_cols]
                 .rename(columns={"Cluster_Risk_pct": "% Risk Contribution"}),
        use_container_width=True
    )

    st.caption(
        f"Rows={Y.shape[0]} | Tickers={Y.shape[1]} | k={labels.max()} | "
        f"rm={rm} α={alpha:.2f} | weights=gross (|w|=1)"
    )