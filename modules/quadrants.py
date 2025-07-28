import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ---------------- Shared window map (match risk_metrics_tables) ----------------
_WINDOW_DAYS = {'1M':21,'3M':63,'6M':126,'1Y':252}

# ---------------- Utilities reused from metrics page (SAFE COPIES) -------------
def _clean_pct_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert '12.34%' strings back to float 12.34 leaving non-percent cells unchanged."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object and col != "Security Name":
            df[col] = df[col].apply(
                lambda v: float(v.rstrip('%')) if isinstance(v, str) and v.endswith('%') else v
            )
    return df

def _prepare_plot_df(tables: dict, portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Merge the three metrics tables and portfolio metadata into a single numeric frame."""
    desc_clean = _clean_pct_df(tables['descriptive'])
    tail_clean = _clean_pct_df(tables['tail_risk'])
    draw_clean = _clean_pct_df(tables['drawdowns'])

    tail_cols_needed = ["Security Name","VaR95","CVaR95","EVaR95","RLVaR95","TG95","WR","LPM1","LPM2"]
    tail_cols = [c for c in tail_cols_needed if c in tail_clean.columns]

    draw_cols_needed = ["Security Name","MDD_Abs","ADD_Abs","DaR_Abs95","CDaR_Abs95","UCI_Abs"]
    draw_cols = [c for c in draw_cols_needed if c in draw_clean.columns]

    plot_df = (
        desc_clean[["Security Name","MAD","SemiDev","Weight"]]
        .merge(tail_clean[tail_cols], on="Security Name", how="left")
        .merge(draw_clean[draw_cols], on="Security Name", how="left")
    )

    if 'EOD Name' in portfolio_df.columns:
        meta = (portfolio_df
                .rename(columns={'EOD Name':'Security Name'})
                .set_index('Security Name'))
        meta_cols = [c for c in ['EOD Ticker','Sector','Category',
                                 'Long_weight','Short_weight','Net_weight']
                     if c in meta.columns]
        plot_df = (plot_df
                   .set_index('Security Name')
                   .join(meta[meta_cols], how='left')
                   .reset_index())
    else:
        for c in ['EOD Ticker','Sector','Category','Long_weight','Short_weight','Net_weight']:
            if c not in plot_df.columns:
                plot_df[c] = np.nan

    for c in plot_df.columns:
        if c not in ('Security Name','EOD Ticker','Sector','Category'):
            plot_df[c] = pd.to_numeric(plot_df[c], errors='ignore')
    return plot_df

def _scatter_quadrant(df: pd.DataFrame, x_col: str, y_col: str,
                      color_mode: str, size_mode: str = "abs_net_weight",
                      title: str = ""):
    df = df.copy()

    # size
    if size_mode == "abs_net_weight" and "Net_weight" in df.columns:
        df["BubbleSize"] = df["Net_weight"].abs() * 100
    else:
        df["BubbleSize"] = 10.0

    # color
    color_arg = None
    color_map = None
    cm = (color_mode or "").lower()
    if cm == "all" and "Category" in df.columns:
        df["HL"] = df["Category"].str.title()
        color_arg, color_map = "HL", {"Long":"limegreen","Short":"crimson"}
    elif cm == "sector" and "Sector" in df.columns:
        color_arg = "Sector"
    elif cm == "long" and "Category" in df.columns:
        df["HL"] = np.where(df["Category"].str.upper()=="LONG","Long","Other")
        color_arg, color_map = "HL", {"Long":"limegreen","Other":"lightgray"}
    elif cm == "short" and "Category" in df.columns:
        df["HL"] = np.where(df["Category"].str.upper()=="SHORT","Short","Other")
        color_arg, color_map = "HL", {"Short":"crimson","Other":"lightgray"}

    fig = px.scatter(
        df, x=x_col, y=y_col,
        size="BubbleSize", color=color_arg,
        color_discrete_map=color_map,
        hover_name="Security Name",
        labels={x_col:f"{x_col} (%)", y_col:f"{y_col} (%)"},
        title=title or f"{y_col} vs {x_col}",
        template="plotly_dark",
        size_max=25
    )

    # median lines
    fig.add_vline(x=df[x_col].median(), line_dash="dash", line_color="grey")
    fig.add_hline(y=df[y_col].median(), line_dash="dash", line_color="grey")

    fig.update_layout(
        legend_title_text=color_mode if color_mode and color_mode != "None" else "",
        margin=dict(l=40,r=20,t=70,b=50)
    )
    return fig

# ---------------- PUBLIC ENTRYPOINT -------------------------------------------
def render_quadrants(master_df, longs_df, shorts_df, portfolio_df,
                     build_and_split, page_header):
    page_header("","Quadrants","Scatter & quadrant view of risk metrics")

    # ---- Book & Window controls ----
    with st.sidebar:
        st.markdown("**Quadrants – Data Scope**")
        book_sel   = st.multiselect("Book", ["All","Long","Short"],
                                    default=["All"], max_selections=1)
        window_sel = st.multiselect("Look-back", list(_WINDOW_DAYS.keys()),
                                    default=["1Y"], max_selections=1)
    if not book_sel or not window_sel:
        st.info("Select Book and Look-back.")
        return
    book   = book_sel[0]
    window = window_sel[0]

    # map to returns
    base_returns = {
        'All': master_df,
        'Long': longs_df,
        'Short': shorts_df
    }[book]

    # ---- Compute or fetch metrics ----
    cache_key = (book, window)
    tables = st.session_state.get('risk_tables_cache', {}).get(cache_key)
    if tables is None:
        try:
            tables = build_and_split(base_returns, book if book!="All" else "FullHist", portfolio_df, window)
        except Exception as e:
            st.error(f"Failed to compute risk tables for Quadrants: {e}")
            return
        st.session_state.setdefault('risk_tables_cache', {})[cache_key] = tables

    # ---- Prepare plot df ----
    try:
        plot_df = _prepare_plot_df(tables, portfolio_df)
    except Exception as e:
        st.error(f"Failed to prepare plot data: {e}")
        return

    # ---- Plot controls ----
    exclude = {'Security Name','EOD Ticker','Sector','Category',
               'Long_weight','Short_weight','Net_weight','Weight'}
    numeric = [c for c in plot_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(plot_df[c])]
    with st.sidebar:
        st.markdown("**Quadrants – Plot Controls**")
        x_col = st.selectbox("X Metric", numeric, index=0)
        y_col = st.selectbox("Y Metric", numeric, index= numeric.index("CVaR95") if "CVaR95" in numeric else 1)
        color_choice = st.selectbox("Color By", ["None","All","Sector","Long","Short"], index=0)
        sectors = sorted(plot_df["Sector"].dropna().unique())
        selected_sectors = st.multiselect("Filter Sectors", sectors, default=sectors)
        min_w = st.slider("Min |Net Weight| %", 0.0, float(plot_df["Net_weight"].abs().max()*100), 0.0, 0.5)

    # ---- Apply filters ----
    plot_df = plot_df[plot_df["Sector"].isin(selected_sectors)]
    plot_df = plot_df[plot_df["Net_weight"].abs()*100 >= min_w]
    if plot_df.empty:
        st.info("No securities match current filters.")
        return

    # ---- Render scatter ----
    fig = _scatter_quadrant(
        plot_df, x_col, y_col,
        None if color_choice=="None" else color_choice.lower(),
        size_mode="abs_net_weight",
        title=f"{y_col} vs {x_col}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Underlying data ----
    with st.expander("Underlying Data"):
        base_cols = ["Security Name","Sector","Category","Net_weight", x_col, y_col]
        extra     = [c for c in numeric if c not in base_cols]
        st.dataframe(plot_df[base_cols+extra].sort_values(by=y_col, ascending=False),
                     use_container_width=True)

    st.caption(f"{book} | {window} | rows={base_returns.shape[0]} | tickers={base_returns.shape[1]}")