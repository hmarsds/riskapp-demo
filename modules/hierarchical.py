import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st

def render_hierarchical(
    master_df: pd.DataFrame,
    longs_df: pd.DataFrame,
    shorts_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    page_header
):
    # 0. Header
    page_header("", "Hierarchical Clustering", "Spearman clusters by book & look-back")

    # 1. Sidebar controls
    book   = st.sidebar.selectbox("Book", ["All", "Long", "Short"], index=0)
    window = st.sidebar.selectbox("Window", ["6M", "1Y", "3Y", "5Y"], index=1)
    days_map = {"6M":126, "1Y":252, "3Y":756, "5Y":1260}

    # 2. Select and slice returns
    df_map = {"All": master_df, "Long": longs_df, "Short": shorts_df}
    df = df_map[book].copy()
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    df.index = pd.to_datetime(df.index)
    df = df.iloc[-days_map[window]:]

    # 3. Clean returns
    def preprocess_returns(x):
        cleaned = x.dropna(axis=1, thresh=int(0.75 * len(x)))
        return cleaned.ffill().bfill(limit=5)
    Y = preprocess_returns(df)
    if Y.shape[1] < 2:
        st.warning("Not enough assets after cleaning for clustering.")
        return

    # 4. Compute Spearman corr, distance, linkage
    codep    = Y.corr(method="spearman")
    dist_mat = np.sqrt(0.5 * (1 - codep))
    condensed= sch.distance.squareform(dist_mat.values)
    linkage  = sch.linkage(condensed, method="ward", optimal_ordering=True)

    # 5. Auto‑k via two‑difference gap
    def find_k(link, max_k=10, min_size=2):
        heights = link[:,2]
        top_n   = np.sort(heights)[-max_k-1:]
        diffs   = np.diff(top_n)
        ddiffs  = diffs[1:] - diffs[:-1]
        best    = np.argmax(ddiffs) + 1
        n_leaves= link.shape[0] + 1
        k0      = n_leaves - (best + 1)
        for k in range(k0, 1, -1):
            lbls = sch.fcluster(link, t=k, criterion="maxclust")
            if np.bincount(lbls)[1:].min() >= min_size:
                return k
        return 2
    max_k    = min(10, codep.shape[1]-1)
    k        = find_k(linkage, max_k=max_k)
    clusters = sch.fcluster(linkage, t=k, criterion="maxclust")

    # 6. Reorder leaves
    leaves    = sch.dendrogram(linkage, no_plot=True)["leaves"]
    codep_ord = codep.iloc[leaves, leaves]
    tickers   = codep_ord.columns.tolist()

    # === Interactive Plotly heatmap ===
    fig = go.Figure(go.Heatmap(
        z=codep_ord.values,
        x=tickers,
        y=tickers,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        colorbar=dict(title="Spearman ρ")
    ))
    fig.update_xaxes(tickangle=90, automargin=True)
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_layout(
        width=800, height=800,
        margin=dict(l=120, r=20, t=50, b=120)
    )
    permuted = [clusters[i] for i in leaves]
    spans = {}
    for idx, cl in enumerate(permuted):
        spans.setdefault(cl, [idx, idx])[1] = idx
    for start, end in spans.values():
        if end > start:
            fig.add_shape(
                type="rect",
                x0=start-0.5, x1=end+0.5,
                y0=start-0.5, y1=end+0.5,
                line=dict(color="magenta", width=3),
                fillcolor="rgba(0,0,0,0)"
            )
    st.subheader("Interactive Spearman Clustermap")
    st.plotly_chart(fig, use_container_width=True)

    # === Static dendrogram (dark) ===
    plt.style.use("dark_background")
    heights_sorted = np.sort(linkage[:,2])
    threshold = heights_sorted[-(k-1)] if k>1 else 0
    fig2, ax2 = plt.subplots(figsize=(14,5))
    sch.dendrogram(
        linkage,
        labels=tickers,
        leaf_rotation=90,
        color_threshold=threshold,
        above_threshold_color="grey",
        ax=ax2
    )
    ax2.set_title(f"Dendrogram (k={k})", color="white")
    ax2.set_ylabel("Distance", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("white")
    st.subheader("Dendrogram")
    st.pyplot(fig2)

    # === Cluster Membership table (original format) ===
    st.subheader("Cluster Memberships")
    rows = []
    sector_map = portfolio_df.set_index("EOD Ticker")["Sector"].to_dict()
    for cl in sorted(set(clusters)):
        members = [tickers[i] for i,v in enumerate(clusters) if v==cl]
        sub = codep.loc[members, members]
        avg_rho = sub.values[np.triu_indices_from(sub,1)].mean() if len(members)>1 else np.nan
        secs = sector_map.get(members[0], "") if members else ""
        rows.append({
            "Cluster":   cl,
            "Tickers":   ", ".join(members),
            "Avg ρ":      round(avg_rho, 3),
            "Sector(s)": ", ".join(sorted({sector_map.get(t,"") for t in members})),
            "Book":      book
        })
    df_mem = pd.DataFrame(rows)
    st.dataframe(df_mem, use_container_width=True)

    # === Top 20 Intra‑Cluster Correlations Across All Clusters ===
    st.subheader("Top 20 Intra‑Cluster Correlations")
    rows = []
    sector_map = portfolio_df.set_index("EOD Ticker")["Sector"].to_dict()
    from itertools import combinations

    # collect every intra‐cluster pair
    for cl in sorted(set(clusters)):
        members = [tickers[i] for i, v in enumerate(clusters) if v == cl]
        if len(members) < 2:
            continue
        subρ = codep.loc[members, members]
        for t1, t2 in combinations(members, 2):
            rows.append({
                "Cluster":     cl,
                "Ticker 1":    t1,
                "Ticker 2":    t2,
                "Correlation": subρ.at[t1, t2],
                "Sector 1":    sector_map.get(t1, "Unknown"),
                "Sector 2":    sector_map.get(t2, "Unknown"),
                "Book":        book
            })

    # build DataFrame, sort, take top 20
    df_pairs = (
        pd.DataFrame(rows)
          .sort_values("Correlation", ascending=False)
          .head(20)
          .reset_index(drop=True)
    )

    st.dataframe(df_pairs, use_container_width=True)