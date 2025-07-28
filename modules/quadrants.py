# modules/quadrants.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ---------------- Shared window map ----------------
_WINDOW_DAYS = {'1M': 21, '3M': 63, '6M': 126, '1Y': 252}

# ---------------- Utilities ----------------
def _clean_pct_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object and col != "Security Name":
            df[col] = df[col].apply(
                lambda v: float(v.rstrip('%')) if isinstance(v, str) and v.endswith('%') else v
            )
    return df

def _prepare_plot_df(tables: dict, portfolio_df: pd.DataFrame) -> pd.DataFrame:
    desc = _clean_pct_df(tables['descriptive'])
    tail = _clean_pct_df(tables['tail_risk'])
    draw = _clean_pct_df(tables['drawdowns'])

    plot_df = (
        desc.merge(tail, on="Security Name", how="left")
            .merge(draw, on="Security Name", how="left")
    )

    # bring in portfolio metadata
    meta = (
        portfolio_df
        .rename(columns={'EOD Name': 'Security Name'})
        .set_index('Security Name')
    )
    join_cols = ['EOD Ticker','Sector','Category','Long_weight','Short_weight','Net_weight']
    plot_df = (
        plot_df
        .set_index('Security Name')
        .join(meta[join_cols], how='left')
        .reset_index()
    )

    # coerce numerics
    for c in plot_df.columns:
        if c not in ('Security Name','EOD Ticker','Sector','Category'):
            plot_df[c] = pd.to_numeric(plot_df[c], errors='ignore')

    return plot_df

def _scatter_quadrant(df, x_col, y_col, color_mode, size_mode="abs_net_weight", title=""):
    df = df.copy()
    df["BubbleSize"] = df["Net_weight"].abs() * 100 if "Net_weight" in df.columns else 10

    cm = (color_mode or "").lower()
    if cm == "all" and "Category" in df:
        df["HL"] = df["Category"].str.title()
        color_arg, cmap = "HL", {"Long":"limegreen","Short":"crimson"}
    elif cm == "sector" and "Sector" in df:
        color_arg, cmap = "Sector", None
    elif cm == "long" and "Category" in df:
        df["HL"] = np.where(df["Category"].str.upper()=="LONG","Long","Other")
        color_arg, cmap = "HL", {"Long":"limegreen","Other":"lightgray"}
    elif cm == "short" and "Category" in df:
        df["HL"] = np.where(df["Category"].str.upper()=="SHORT","Short","Other")
        color_arg, cmap = "HL", {"Short":"crimson","Other":"lightgray"}
    else:
        color_arg, cmap = None, None

    fig = px.scatter(
        df, x=x_col, y=y_col,
        size="BubbleSize", color=color_arg,
        color_discrete_map=cmap,
        hover_name="Security Name",
        labels={x_col:f"{x_col} (%)", y_col:f"{y_col} (%)"},
        title=title or f"{y_col} vs {x_col}",
        template="plotly_dark", size_max=25
    )

    fig.add_vline(x=df[x_col].median(), line_dash="dash", line_color="grey")
    fig.add_hline(y=df[y_col].median(), line_dash="dash", line_color="grey")
    fig.update_layout(margin=dict(l=40,r=20,t=70,b=50),
                      legend_title_text=color_mode if color_mode!="None" else "")

    return fig

# ---------------- Public Entry Point ----------------
def render_quadrants(master_df, longs_df, shorts_df, portfolio_df,
                     build_and_split, page_header):
    page_header("","Quadrants","Scatter & quadrant view of risk metrics")

    with st.sidebar:
        st.markdown("**Quadrants – Data Scope**")
        book_sel   = st.multiselect("Book", ["All","Long","Short"],
                                    default=["All"], max_selections=1)
        window_sel = st.multiselect("Look-back", list(_WINDOW_DAYS.keys()),
                                    default=["1Y"], max_selections=1)
    if not book_sel or not window_sel:
        st.info("Select Book and Look-back."); return

    book   = book_sel[0]
    window = window_sel[0]

    base_returns = {'All': master_df, 'Long': longs_df, 'Short': shorts_df}[book]

    # compute or fetch cached tables
    cache_key = (book, window)
    tables = st.session_state.get('risk_tables_cache', {}).get(cache_key)
    if tables is None:
        tables = build_and_split(
            base_returns,
            "FullHist" if book=="All" else book,
            portfolio_df,
            window
        )
        st.session_state.setdefault('risk_tables_cache', {})[cache_key] = tables

    # prepare merged DataFrame
    plot_df = _prepare_plot_df(tables, portfolio_df)

    # ---- Book-based filter ----
    if book == "Long":
        plot_df = plot_df[plot_df["Long_weight"] > 0]
    elif book == "Short":
        plot_df = plot_df[plot_df["Short_weight"] > 0]
    else:  # All
        plot_df = plot_df[(plot_df["Long_weight"]!=0) | (plot_df["Short_weight"]!=0)]

    # ---- Plot controls ----
    exclude = {'Security Name','EOD Ticker','Sector','Category',
               'Long_weight','Short_weight','Net_weight','Weight'}
    numeric = [c for c in plot_df.columns
               if c not in exclude and pd.api.types.is_numeric_dtype(plot_df[c])]

    with st.sidebar:
        st.markdown("**Quadrants – Plot Controls**")
        x_col = st.selectbox("X Metric", numeric, index=0)
        default_y = "CVaR95" if "CVaR95" in numeric else numeric[min(1,len(numeric)-1)]
        y_col = st.selectbox("Y Metric", numeric, index=numeric.index(default_y))
        color_choice = st.selectbox("Color By", ["None","All","Sector","Long","Short"], index=0)
        sectors = sorted(plot_df["Sector"].dropna().unique())
        selected_sectors = st.multiselect("Filter Sectors", sectors, default=sectors)
        min_w = st.slider(
            "Min |Net Weight| %",
            0.0,
            float(plot_df["Net_weight"].abs().max()*100),
            0.0,
            0.5,
        )

    # apply sector & weight filters
    plot_df = plot_df[plot_df["Sector"].isin(selected_sectors)]
    plot_df = plot_df[plot_df["Net_weight"].abs()*100 >= min_w]
    if plot_df.empty:
        st.info("No securities match current filters."); return

    # render
    fig = _scatter_quadrant(
        plot_df, x_col, y_col,
        None if color_choice=="None" else color_choice,
        size_mode="abs_net_weight",
        title=f"{y_col} vs {x_col}"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Underlying Data"):
        cols = ["Security Name","Sector","Category","Net_weight", x_col, y_col]
        extra = [c for c in numeric if c not in cols]
        st.dataframe(plot_df[cols+extra].sort_values(by=y_col, ascending=False),
                     use_container_width=True)

    st.caption(f"{book} | {window} | rows={base_returns.shape[0]} | tickers={base_returns.shape[1]}")