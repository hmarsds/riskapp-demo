# modules/centrality.py
# -------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

# -------------------------- Constants -------------------------- #
_LOOKBACK = {"6M": 126, "1Y": 252, "3Y": 756}
_METRICS  = ["Degree", "Betweenness", "Closeness", "Eigenvector", "PageRank"]

# -------------------------- Helpers -------------------------- #
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
    """MST on distance = 1-|rho|. Store original corr on edges."""
    dist = 1 - corr.abs().values
    np.fill_diagonal(dist, 0)
    G_full = nx.Graph()
    ticks = corr.index.tolist()
    for i, u in enumerate(ticks):
        for j in range(i + 1, len(ticks)):
            v = ticks[j]
            G_full.add_edge(u, v, weight=dist[i, j], corr=corr.loc[u, v])
    return nx.minimum_spanning_tree(G_full, weight="weight")

def _centralities(G):
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    clo = nx.closeness_centrality(G)
    eig = nx.eigenvector_centrality_numpy(G)
    pr  = nx.pagerank(G)
    return pd.DataFrame({
        "Ticker": list(G.nodes()),
        "Degree": pd.Series(deg),
        "Betweenness": pd.Series(btw),
        "Closeness": pd.Series(clo),
        "Eigenvector": pd.Series(eig),
        "PageRank": pd.Series(pr)
    })

def _layout_positions(G, layout_choice):
    if layout_choice == "Spring (force-directed)":
        return nx.spring_layout(G, iterations=500, seed=42)
    if layout_choice == "Kamada-Kawai":
        return nx.kamada_kawai_layout(G, weight="weight")
    if layout_choice == "Circular":
        return nx.circular_layout(G)
    if layout_choice == "Random":
        return nx.random_layout(G, seed=42)
    # Fallback
    return nx.spring_layout(G, iterations=500, seed=42)

def _book_class(w):
    if w > 0:  return "Long"
    if w < 0:  return "Short"
    return "Flat"

def _hover_tmpl(show_labels):
    # Bold header: if labels shown, it's %{text} (Ticker); otherwise use customdata[0] (Ticker)
    head = "<b>%{text}</b><br>" if show_labels else "<b>%{customdata[0]}</b><br>"
    return (head +
            "Name: %{customdata[1]}<br>" +
            "Sector: %{customdata[2]}<br>" +
            "Book: %{customdata[3]}<br>" +
            "Gross wt: %{customdata[4]:.2%}<br>" +
            "Deg: %{customdata[5]:.3f} | Btw: %{customdata[6]:.3f}<br>" +
            "Clo: %{customdata[7]:.3f} | Eig: %{customdata[8]:.3f}<br>" +
            "PR: %{customdata[9]:.3f}<extra></extra>")

def _customdata(df, include_ticker=True):
    """Builds the customdata matrix safely even if some cols are missing."""
    ticker_vals = df["Ticker"].values
    name_vals   = df.get("Name", df.get("EOD Name", df["Ticker"])).values
    sector_vals = df.get("Sector", pd.Series(["Unknown"] * len(df))).values
    book_vals   = df.get("Book",   pd.Series(["Unknown"] * len(df))).values
    gw_vals     = df.get("GrossWeight", pd.Series([0.0] * len(df))).values

    return np.stack([
        ticker_vals if include_ticker else np.zeros(len(df), dtype=object),
        name_vals,
        sector_vals,
        book_vals,
        gw_vals,
        df["Degree"].values,
        df["Betweenness"].values,
        df["Closeness"].values,
        df["Eigenvector"].values,
        df["PageRank"].values
    ], axis=-1)

def _add_cat_trace(fig, subset, sizes, show_labels, label_size, color, name):
    fig.add_trace(go.Scatter(
        x=subset["x"], y=subset["y"],
        mode="markers+text" if show_labels else "markers",
        text=subset["Ticker"] if show_labels else None,   # keep ticker text
        textposition="top center",
        textfont=dict(size=label_size),
        marker=dict(
            size=sizes[subset.index],
            color=color,
            line=dict(width=0.5, color="rgba(255,255,255,0.4)")
        ),
        name=name,
        hovertemplate=_hover_tmpl(show_labels),
        customdata=_customdata(subset, include_ticker=not show_labels)
    ))

def _centrality_heatmap(df_cent):
    """
    df_cent: index = ticker, columns = metrics. We want metrics on y (rows).
    """
    mat = df_cent.T  # rows: metrics, cols: tickers
    fig = px.imshow(
        mat.values,
        x=mat.columns,
        y=mat.index,
        color_continuous_scale="Plasma",
        aspect="auto",
        labels=dict(color="Value")
    )
    fig.update_xaxes(tickangle=-90, tickmode="array",
                     tickvals=list(range(len(mat.columns))), ticktext=mat.columns)
    fig.update_yaxes(tickmode="array",
                     tickvals=list(range(len(mat.index))), ticktext=mat.index)
    fig.update_layout(template="plotly_dark", margin=dict(l=60, r=40, t=40, b=80))
    return fig

# -------------------------- Sidebar -------------------------- #
def _controls():
    st.sidebar.markdown("**Network Controls**")
    book   = st.sidebar.selectbox("Book", ["All", "Long", "Short"], 0)
    window = st.sidebar.selectbox("Look-back", list(_LOOKBACK.keys()), 1)
    color_by = st.sidebar.selectbox(
        "Color nodes by",
        ["Sector", "Book"] + _METRICS,
        index=0
    )
    size_mode = st.sidebar.radio("Node size by", ["Equal", "Gross weight"], index=0)
    eq_size   = st.sidebar.slider("Equal node size", 8, 40, 18)
    show_lab  = st.sidebar.checkbox("Show ticker labels", value=True)
    lab_size  = st.sidebar.slider("Label font size", 8, 22, 12)
    edge_alpha= st.sidebar.slider("Edge opacity", 0.1, 1.0, 0.35, 0.05)
    layout_choice = st.sidebar.selectbox(
        "Network layout",
        ["Spring (force-directed)", "Kamada-Kawai", "Circular", "Random"],
        index=0
    )
    return (book, window, color_by, size_mode, eq_size,
            show_lab, lab_size, edge_alpha, layout_choice)

# -------------------------- Main render -------------------------- #
def render_network(master_df, longs_df, shorts_df, portfolio_df, page_header):
    """
    portfolio_df must contain: 'EOD Ticker', 'Net_weight', and ideally 'EOD Name'/'Name', 'Sector', 'Book'.
    If missing, fallbacks are applied.
    """
    page_header("", "Centrality / Correlation Network (MST)",
                "Edges are MST of |Spearman ρ|; width ∝ |ρ|. Choose color/size & layout.")

    # controls
    (book, window, color_by, size_mode, eq_size,
     show_labels, label_size, edge_alpha, layout_choice) = _controls()

    # slice data
    Y = _slice_returns(master_df, window, book, longs_df, shorts_df)
    if Y.empty:
        st.warning("No return data after filtering.")
        return

    corr = Y.corr(method="spearman")
    G = _mst_from_corr(corr)
    cent_df = _centralities(G)

    # ---- Meta & weights ----
    meta_cols = [c for c in ["EOD Ticker", "EOD Name", "Name", "Sector", "Book"] if c in portfolio_df.columns]
    if "EOD Ticker" not in meta_cols:
        st.error("portfolio_df must include 'EOD Ticker'.")
        return
    meta_keep = portfolio_df[meta_cols].drop_duplicates("EOD Ticker")

    gross_w = portfolio_df.set_index("EOD Ticker")["Net_weight"].abs()
    net_w   = portfolio_df.set_index("EOD Ticker")["Net_weight"]

    df_nodes = (cent_df.set_index("Ticker")
                        .join(meta_keep.set_index("EOD Ticker"), how="left", rsuffix="_pf")
                        .join(gross_w.rename("GrossWeight"), how="left")
                        .join(net_w.rename("NetWeight"), how="left")
                        .reset_index())

    # ---------- NORMALISE COLUMN NAMES ----------
    # Ensure df_nodes has "Name"
    if "Name" not in df_nodes.columns:
        if "EOD Name" in df_nodes.columns:
            df_nodes.rename(columns={"EOD Name": "Name"}, inplace=True)
        else:
            name_src = "EOD Name" if "EOD Name" in portfolio_df.columns else ("Name" if "Name" in portfolio_df.columns else None)
            if name_src:
                name_map = portfolio_df.set_index("EOD Ticker")[name_src]
                df_nodes["Name"] = df_nodes["Ticker"].map(name_map)
            else:
                df_nodes["Name"] = df_nodes["Ticker"]
    df_nodes["Name"] = df_nodes["Name"].fillna(df_nodes["Ticker"])

    # Ensure Sector exists
    if "Sector" not in df_nodes.columns:
        if "Sector" in portfolio_df.columns:
            sec_map = portfolio_df.set_index("EOD Ticker")["Sector"]
            df_nodes["Sector"] = df_nodes["Ticker"].map(sec_map)
        else:
            df_nodes["Sector"] = "Unknown"
    df_nodes["Sector"] = df_nodes["Sector"].fillna("Unknown")

    # Book classification if not present
    if "Book" not in df_nodes.columns:
        df_nodes["Book"] = df_nodes["NetWeight"].apply(_book_class)
    else:
        df_nodes["Book"] = df_nodes["Book"].fillna(df_nodes["NetWeight"].apply(_book_class))

    # sizes
    if size_mode == "Gross weight":
        if df_nodes["GrossWeight"].max() == 0:
            sizes = np.repeat(eq_size, len(df_nodes))
        else:
            norm = df_nodes["GrossWeight"] / df_nodes["GrossWeight"].max()
            sizes = (norm * 35 + 8).values
    else:
        sizes = np.repeat(eq_size, len(df_nodes))

    # positions
    pos = _layout_positions(G, layout_choice)
    df_nodes["x"] = df_nodes["Ticker"].map(lambda t: pos[t][0])
    df_nodes["y"] = df_nodes["Ticker"].map(lambda t: pos[t][1])

    # edges
    edge_x, edge_y, edge_w = [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
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

    # ----------- Build network fig ----------- #
    fig = go.Figure()
    fig.add_trace(edge_trace)

    if color_by in ["Sector", "Book"]:
        if color_by == "Sector":
            palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Set3
            cats = df_nodes["Sector"].unique()
            cmap = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
        else:  # Book
            cmap = {"Long": "#2ca02c", "Short": "#d62728", "Flat": "#aaaaaa"}
            cats = ["Long", "Short", "Flat"]

        for cat in cats:
            subset = df_nodes[df_nodes[color_by] == cat]
            if subset.empty:
                continue
            _add_cat_trace(fig, subset, sizes, show_labels, label_size,
                           cmap[cat], name=cat)
        fig.update_layout(showlegend=True)

    else:
        vals = df_nodes[color_by].values
        fig.add_trace(go.Scatter(
            x=df_nodes["x"], y=df_nodes["y"],
            mode="markers+text" if show_labels else "markers",
            text=df_nodes["Ticker"] if show_labels else None,   # ticker labels
            textposition="top center",
            textfont=dict(size=label_size),
            marker=dict(
                size=sizes,
                color=vals,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title=color_by),
                line=dict(width=0.5, color="rgba(255,255,255,0.4)")
            ),
            hovertemplate=_hover_tmpl(show_labels),
            customdata=_customdata(df_nodes, include_ticker=not show_labels)
        ))
        fig.update_layout(showlegend=False)

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        dragmode="pan"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----------- Heatmap ----------- #
    st.markdown("### Centrality Heatmap")
    cent_mat = (df_nodes[["Ticker"] + _METRICS]
                .set_index("Ticker")
                .sort_index())
    fig_hm = _centrality_heatmap(cent_mat)
    st.plotly_chart(fig_hm, use_container_width=True)

    # ----------- Table ----------- #
    st.markdown("### Centrality Table")
    # Make a display copy so we don't pollute df_nodes
    display_df = df_nodes.copy()
    display_df["GrossWeightPct"] = display_df["GrossWeight"] * 100

    table_cols = ["Ticker", "Name", "Sector", "Book", "GrossWeightPct"] + _METRICS
    st.dataframe(
        display_df[table_cols].sort_values("Ticker").reset_index(drop=True),
        use_container_width=True,
        column_config={
            "GrossWeightPct": st.column_config.NumberColumn("Gross wt (%)", format="%.2f")
        }
    )