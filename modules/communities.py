# modules/communities.py
# -------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

# -------------------------- Constants -------------------------- #
_LOOKBACK = {"6M": 126, "1Y": 252, "3Y": 756}

# -------------------------- Data prep -------------------------- #
def _slice_returns(df, window, book, longs_df, shorts_df):
    X = df.copy()
    if "date" in X.columns:
        X = X.drop(columns="date")
    X.index = pd.to_datetime(X.index)
    X = X.iloc[-_LOOKBACK[window]:]

    if book == "Long":
        X = X.reindex(columns=longs_df.columns)
    elif book == "Short":
        X = X.reindex(columns=shorts_df.columns)

    X = X.dropna(axis=1, thresh=int(0.75 * len(X))).ffill().bfill(limit=5)
    return X

def _mst_from_corr(corr):
    """Minimum Spanning Tree on distance = 1 - |rho| (absolute Spearman)."""
    dist = 1 - corr.abs().values
    np.fill_diagonal(dist, 0)
    G_full = nx.Graph()
    ticks = corr.index.tolist()
    for i, u in enumerate(ticks):
        for j in range(i + 1, len(ticks)):
            v = ticks[j]
            G_full.add_edge(u, v, weight=dist[i, j], corr=corr.loc[u, v])
    return nx.minimum_spanning_tree(G_full, weight="weight")

def _layout_positions(G, layout_kind):
    if layout_kind == "Spring (force)":
        return nx.spring_layout(G, iterations=500, seed=42)
    if layout_kind == "Kamada-Kawai":
        return nx.kamada_kawai_layout(G)
    if layout_kind == "Circular":
        return nx.circular_layout(G)
    return nx.spring_layout(G, iterations=500, seed=42)

def _book_class(w):
    if w > 0:  return "Long"
    if w < 0:  return "Short"
    return "Flat"

# ---------------------- Plot helpers ---------------------- #
def _hover_tmpl(show_labels):
    # 0 ticker, 1 name, 2 sector, 3 book, 4 gross, 10 comm, 11 size, 12 avg|ρ|
    head = "<b>%{text}</b><br>" if show_labels else "<b>%{customdata[0]}</b><br>"
    return (head +
            "Name: %{customdata[1]}<br>" +
            "Sector: %{customdata[2]}<br>" +
            "Book: %{customdata[3]}<br>" +
            "Gross wt: %{customdata[4]:.2%}<br>" +
            "Community: %{customdata[10]}<br>" +
            "Size(comm): %{customdata[11]}<br>" +
            "Avg |ρ|: %{customdata[12]:.3f}<extra></extra>")

def _customdata(df, include_ticker=True):
    # Safe getters
    ticker_vals = df["Ticker"].values
    name_vals   = df.get("Name", df.get("EOD Name", df["Ticker"])).values
    sector_vals = df.get("Sector", pd.Series(["Unknown"] * len(df))).values
    book_vals   = df.get("Book",   pd.Series(["Unknown"] * len(df))).values
    gw_vals     = df.get("GrossWeight", pd.Series([0.0] * len(df))).values

    base = np.column_stack([
        ticker_vals if include_ticker else np.zeros(len(df), dtype=object),  # 0
        name_vals,                                                           # 1
        sector_vals,                                                         # 2
        book_vals,                                                           # 3
        gw_vals                                                              # 4
    ])
    zeros_5_9 = np.zeros((df.shape[0], 5))  # 5..9 unused
    tail = np.column_stack([
        df.get("Community", pd.Series([-1]*len(df))).values,        # 10
        df.get("Comm_Size", pd.Series([0]*len(df))).values,          # 11
        df.get("Comm_AvgAbsCorr", pd.Series([0.0]*len(df))).values   # 12
    ])
    return np.concatenate([base, zeros_5_9, tail], axis=1)

def _add_cat_trace(fig, subset, sizes_vec, show_labels, label_size, color, name):
    fig.add_trace(go.Scatter(
        x=subset["x"], y=subset["y"],
        mode="markers+text" if show_labels else "markers",
        text=subset["Ticker"] if show_labels else None,  # keep tickers
        textposition="top center",
        textfont=dict(size=label_size),
        marker=dict(
            size=sizes_vec[subset.index],
            color=color,
            line=dict(width=0.5, color="rgba(255,255,255,0.4)")
        ),
        name=name,
        hovertemplate=_hover_tmpl(show_labels),
        customdata=_customdata(subset, include_ticker=not show_labels)
    ))

def _community_heatmap(df_nodes):
    """Binary membership heatmap (rows=communities, cols=tickers)."""
    comm_ids = sorted(df_nodes["Community"].unique())
    mat = pd.DataFrame(0, index=[f"C{c}" for c in comm_ids], columns=df_nodes["Ticker"])
    for _, r in df_nodes.iterrows():
        mat.loc[f"C{r['Community']}", r["Ticker"]] = 1

    fig = px.imshow(
        mat,
        color_continuous_scale=["#111111", "#00ccff"],
        aspect="auto",
        labels=dict(color="Member"),
        zmin=0, zmax=1
    )
    fig.update_layout(template="plotly_dark",
                      margin=dict(l=40, r=40, t=40, b=40))
    fig.update_xaxes(tickangle=-90)
    return fig

# ---------------------- Sidebar controls ---------------------- #
def _controls():
    st.sidebar.markdown("**Communities Controls**")
    book        = st.sidebar.selectbox("Book", ["All", "Long", "Short"], 0)
    window      = st.sidebar.selectbox("Look-back", list(_LOOKBACK.keys()), 1)
    algo        = st.sidebar.selectbox(
        "Community algo",
        ["Louvain (modularity)", "Greedy modularity", "Label propagation", "Girvan–Newman"],
        0
    )
    layout_kind = st.sidebar.selectbox("Layout", ["Spring (force)", "Kamada-Kawai", "Circular"], 0)
    color_by    = st.sidebar.selectbox("Color nodes by", ["Community", "Sector", "Book"], 0)
    size_mode   = st.sidebar.radio("Node size by", ["Equal", "Gross weight"], 0)
    eq_size     = st.sidebar.slider("Equal node size", 8, 40, 18)
    show_lab    = st.sidebar.checkbox("Show ticker labels", value=True)
    lab_size    = st.sidebar.slider("Label font size", 8, 22, 12)
    edge_alpha  = st.sidebar.slider("Edge opacity", 0.1, 1.0, 0.35, 0.05)
    return (book, window, algo, layout_kind, color_by,
            size_mode, eq_size, show_lab, lab_size, edge_alpha)

# ---------------------- Community detection ---------------------- #
def _detect_communities(G, algo):
    if algo.startswith("Louvain"):
        comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42)
    elif algo.startswith("Greedy"):
        comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    elif algo.startswith("Label"):
        comms = nx.algorithms.community.asyn_lpa_communities(G, weight="weight", seed=42)
        comms = list(comms)
    else:  # Girvan–Newman
        gen = nx.algorithms.community.girvan_newman(G)
        n = G.number_of_nodes()
        target = max(2, int(np.sqrt(n)))
        comms = None
        for level in gen:
            comms = level
            if len(level) >= target:
                break
    comm_map = {}
    for i, c in enumerate(comms, start=1):
        for n in c:
            comm_map[n] = i
    return comm_map, comms

# -------------------------- Main render ------------------------- #
def render_communities(master_df, longs_df, shorts_df, portfolio_df, page_header, meta_df=None):
    """
    meta_df optional. If omitted, Name/Sector/Book are taken from portfolio_df if present,
    otherwise filled with defaults.
    """
    page_header("", "Communities (MST + modularity algorithms)",
                "Groups of tightly connected names on the MST of |Spearman ρ|.")

    (book, window, algo, layout_kind, color_by,
     size_mode, eq_size, show_labels, label_size, edge_alpha) = _controls()

    # Slice returns
    Y = _slice_returns(master_df, window, book, longs_df, shorts_df)
    if Y.empty:
        st.warning("No return data after filtering.")
        return

    corr = Y.corr(method="spearman")
    G = _mst_from_corr(corr)

    # Detect communities
    comm_map, comm_sets = _detect_communities(G, algo)

    # Meta & weights
    gross_w = portfolio_df.set_index("EOD Ticker")["Net_weight"].abs()
    net_w   = portfolio_df.set_index("EOD Ticker")["Net_weight"]

    if meta_df is None:
        meta_df = portfolio_df.copy()

    # Normalize Name col in meta_df (EOD Name -> Name)
    if "Name" not in meta_df.columns and "EOD Name" in meta_df.columns:
        meta_df = meta_df.rename(columns={"EOD Name": "Name"})
    if "Name" not in meta_df.columns:
        meta_df["Name"] = "-"

    if "Sector" not in meta_df.columns:
        meta_df["Sector"] = "Unknown"
    if "Book" not in meta_df.columns:
        meta_df["Book"] = meta_df["Net_weight"].apply(_book_class)

    meta_keep = (meta_df[["EOD Ticker", "Name", "Sector", "Book"]]
                 .drop_duplicates("EOD Ticker"))

    df_nodes = (pd.DataFrame({"Ticker": list(G.nodes())})
                  .merge(meta_keep, left_on="Ticker", right_on="EOD Ticker", how="left")
                  .drop(columns=["EOD Ticker"], errors="ignore"))

    # Fallbacks
    df_nodes["Name"] = df_nodes["Name"].fillna(df_nodes["Ticker"])
    df_nodes["Sector"] = df_nodes["Sector"].fillna("Unknown")
    df_nodes["GrossWeight"] = df_nodes["Ticker"].map(gross_w).fillna(0.0)
    df_nodes["NetWeight"]   = df_nodes["Ticker"].map(net_w).fillna(0.0)
    df_nodes["Book"]        = df_nodes["Book"].fillna(df_nodes["NetWeight"].apply(_book_class))
    df_nodes["Community"]   = df_nodes["Ticker"].map(comm_map).astype(int)

    # Community stats
    stats = []
    for cid, members in df_nodes.groupby("Community"):
        mlist = members["Ticker"].tolist()
        sub = corr.loc[mlist, mlist].abs()
        avg_abs = (sub.values[np.triu_indices_from(sub, 1)]).mean() if len(mlist) > 1 else 0.0
        stats.append((cid, len(mlist), avg_abs))
    comm_df = pd.DataFrame(stats, columns=["Community", "Comm_Size", "Comm_AvgAbsCorr"])
    df_nodes = df_nodes.merge(comm_df, on="Community", how="left")

    # Sizes
    if size_mode == "Gross weight":
        if df_nodes["GrossWeight"].max() == 0:
            sizes = np.repeat(eq_size, len(df_nodes))
        else:
            norm = df_nodes["GrossWeight"] / df_nodes["GrossWeight"].max()
            sizes = (norm * 35 + 8).values
    else:
        sizes = np.repeat(eq_size, len(df_nodes))

    # Layout positions
    pos = _layout_positions(G, layout_kind)
    df_nodes["x"] = df_nodes["Ticker"].map(lambda t: pos[t][0])
    df_nodes["y"] = df_nodes["Ticker"].map(lambda t: pos[t][1])

    # Edges
    edge_x, edge_y, edge_w = [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_w.append(abs(d["corr"]))
    ew = np.array(edge_w)
    if ew.size:
        ew = 1 + 6 * (ew - ew.min()) / (ew.max() - ew.min() + 1e-12)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color=f"rgba(255,255,255,{edge_alpha})"),
        hoverinfo="none",
        showlegend=False,
        name=""
    )

    # ----------- Plot network ----------- #
    fig = go.Figure()
    fig.add_trace(edge_trace)

    if color_by == "Community":
        palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Set3
        comm_ids = sorted(df_nodes["Community"].unique())
        cmap = {cid: palette[i % len(palette)] for i, cid in enumerate(comm_ids)}
        for cid in comm_ids:
            sub = df_nodes[df_nodes["Community"] == cid]
            _add_cat_trace(fig, sub, sizes, show_labels, label_size, cmap[cid], f"C{cid}")
        fig.update_layout(showlegend=True)

    elif color_by == "Sector":
        palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Set3
        sectors = df_nodes["Sector"].fillna("Unknown").unique()
        cmap = {sec: palette[i % len(palette)] for i, sec in enumerate(sectors)}
        for sec in sectors:
            sub = df_nodes[df_nodes["Sector"] == sec]
            _add_cat_trace(fig, sub, sizes, show_labels, label_size, cmap[sec], sec)
        fig.update_layout(showlegend=True)

    else:  # Book
        cmap = {"Long": "#2ca02c", "Short": "#d62728", "Flat": "#aaaaaa"}
        for cls in ["Long", "Short", "Flat"]:
            sub = df_nodes[df_nodes["Book"] == cls]
            if sub.empty:
                continue
            _add_cat_trace(fig, sub, sizes, show_labels, label_size, cmap[cls], cls)
        fig.update_layout(showlegend=True)

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        dragmode="pan"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----------- Heatmap ----------- #
    st.markdown("### Community Membership Heatmap")
    st.plotly_chart(_community_heatmap(df_nodes[["Ticker","Community"]]),
                    use_container_width=True)

    # ----------- Summary table ----------- #
    st.markdown("### Community Summary")
    gw_total = df_nodes["GrossWeight"].sum() or 1.0
    comm_sum = (df_nodes.groupby("Community")
                        .agg(Size=("Ticker","count"),
                             GrossWeight=("GrossWeight","sum"),
                             AvgAbsCorr=("Comm_AvgAbsCorr","first"))
                        .reset_index())
    comm_sum["%GrossWt"] = (comm_sum["GrossWeight"]/gw_total*100).round(2)

    tops = (df_nodes.sort_values(["Community","GrossWeight"], ascending=[True,False])
                   .groupby("Community")["Ticker"]
                   .apply(lambda s: ", ".join(s.head(6))))
    comm_sum = comm_sum.merge(tops.rename("Top Members"), on="Community", how="left")

    st.dataframe(
        comm_sum[["Community","Size","%GrossWt","AvgAbsCorr","Top Members"]],
        use_container_width=True
    )

    # ----------- Full membership ----------- #
    st.markdown("### Full Membership Table")
    display_df = df_nodes.copy()
    display_df["GrossWeightPct"] = display_df["GrossWeight"] * 100

    cols = ["Ticker","Name","Sector","Book","GrossWeightPct",
            "Community","Comm_Size","Comm_AvgAbsCorr"]
    st.dataframe(
        display_df[cols].sort_values(["Community","GrossWeightPct"], ascending=[True,False]),
        use_container_width=True,
        column_config={
            "GrossWeightPct": st.column_config.NumberColumn("Gross wt (%)", format="%.2f")
        }
    )