import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ---------------- Shared window map ----------------
_WINDOW_DAYS = {'1M':21,'3M':63,'6M':126,'1Y':252}

# ---------------- Helpers ----------------
def _clean_pct_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert '12.34%' strings back to float 12.34, leave other cells."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object and col != "Security Name":
            df[col] = df[col].apply(
                lambda v: float(v.rstrip('%')) if isinstance(v, str) and v.endswith('%') else v
            )
    return df

def _prepare_plot_df(tables: dict, portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Merge descriptive, tail and drawdown tables with portfolio metadata."""
    desc = _clean_pct_df(tables['descriptive'])
    tail = _clean_pct_df(tables['tail_risk'])
    draw = _clean_pct_df(tables['drawdowns'])

    plot_df = (
        desc[['Security Name','MAD','SemiDev','Weight']]
        .merge(tail, on='Security Name', how='left')
        .merge(draw, on='Security Name', how='left')
    )

    # attach metadata
    if 'EOD Name' in portfolio_df.columns:
        meta = (portfolio_df.rename(columns={'EOD Name':'Security Name'})
                         .set_index('Security Name'))
        cols = [c for c in ['EOD Ticker','Sector','Category',
                            'Long_weight','Short_weight','Net_weight']
                if c in meta.columns]
        plot_df = (plot_df.set_index('Security Name')
                         .join(meta[cols], how='left')
                         .reset_index())
    else:
        for c in ['EOD Ticker','Sector','Category','Long_weight','Short_weight','Net_weight']:
            plot_df[c] = np.nan

    # coerce numerics
    for c in plot_df.columns:
        if c not in ('Security Name','EOD Ticker','Sector','Category'):
            plot_df[c] = pd.to_numeric(plot_df[c], errors='ignore')
    return plot_df

def _scatter_quadrant(df, x_col, y_col, color_mode, size_mode="abs_net_weight", title=""):
    df = df.copy()
    # bubble size
    if size_mode=="abs_net_weight" and "Net_weight" in df:
        df["BubbleSize"] = df["Net_weight"].abs() * 100
    else:
        df["BubbleSize"] = 10

    # coloring
    cm = (color_mode or "").lower()
    if cm=="all" and "Category" in df:
        df["HL"] = df["Category"].str.title()
        color_arg, color_map = "HL", {"Long":"limegreen","Short":"crimson"}
    elif cm=="sector" and "Sector" in df:
        color_arg, color_map = "Sector", None
    elif cm=="long" and "Category" in df:
        df["HL"] = np.where(df["Category"].str.upper()=="LONG","Long","Other")
        color_arg, color_map = "HL", {"Long":"limegreen","Other":"lightgray"}
    elif cm=="short" and "Category" in df:
        df["HL"] = np.where(df["Category"].str.upper()=="SHORT","Short","Other")
        color_arg, color_map = "HL", {"Short":"crimson","Other":"lightgray"}
    else:
        color_arg, color_map = None, None

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
    # medians
    fig.add_vline(x=df[x_col].median(), line_dash="dash", line_color="grey")
    fig.add_hline(y=df[y_col].median(), line_dash="dash", line_color="grey")

    fig.update_layout(
        legend_title_text=color_mode if color_mode and color_mode!="None" else "",
        margin=dict(l=40,r=20,t=70,b=50)
    )
    return fig

# ---------------- PUBLIC ENTRYPOINT ----------------
def render_quadrants(master_df, longs_df, shorts_df, portfolio_df,
                     build_and_split, page_header):
    page_header("","Quadrants","Scatter & quadrant view of risk metrics")

    # Book & Look‑back
    with st.sidebar:
        st.markdown("**Quadrants – Data Scope**")
        book_sel   = st.multiselect("Book", ["All","Long","Short"],
                                    default=["All"], max_selections=1)
        window_sel = st.multiselect("Look‑back", list(_WINDOW_DAYS.keys()),
                                    default=["1Y"], max_selections=1)
    if not book_sel or not window_sel:
        st.info("Select Book and Look‑back."); return
    book, window = book_sel[0], window_sel[0]

    # choose returns
    base = {'All':master_df,'Long':longs_df,'Short':shorts_df}[book]

    # compute or fetch
    cache = st.session_state.setdefault('risk_tables_cache', {})
    key   = (book, window)
    tables = cache.get(key)
    if tables is None:
        try:
            tables = build_and_split(
                base,
                "FullHist" if book=="All" else book,
                portfolio_df,
                window
            )
        except Exception as e:
            st.error(f"Failed to compute risk tables for Quadrants: {e}")
            return
        cache[key] = tables

    # prepare data
    try:
        plot_df = _prepare_plot_df(tables, portfolio_df)
    except Exception as e:
        st.error(f"Failed to prepare plot data: {e}")
        return

    # Plot controls
    # drop any weight/merge residues
    exclude = {'Security Name','EOD Ticker','Sector','Category',
               'Long_weight','Short_weight','Net_weight'}
    exclude |= {c for c in plot_df if c.lower().endswith(('_x','_y'))}
    numeric = [c for c in plot_df if c not in exclude and pd.api.types.is_numeric_dtype(plot_df[c])]

    with st.sidebar:
        st.markdown("**Quadrants – Plot Controls**")
        x_col       = st.selectbox("X Metric", numeric, index=0)
        default_y   = "CVaR95" if "CVaR95" in numeric else numeric[min(1,len(numeric)-1)]
        y_col       = st.selectbox("Y Metric", numeric, index=numeric.index(default_y))
        color       = st.selectbox("Color By", ["None","All","Sector","Long","Short"], index=0)
        sectors     = sorted(plot_df["Sector"].dropna().unique())
        sel_sectors = st.multiselect("Filter Sectors", sectors, default=sectors)

        # slider only if any non-zero net weight
        max_w = plot_df["Net_weight"].abs().max()*100 if "Net_weight" in plot_df else 0
        if max_w>0:
            min_w = st.slider("Min |Net Weight| %", 0.0, float(max_w), 0.0, 0.5)
        else:
            min_w = 0.0

    # apply filters
    plot_df = plot_df[plot_df["Sector"].isin(sel_sectors)]
    if "Net_weight" in plot_df:
        plot_df = plot_df[plot_df["Net_weight"].abs()*100 >= min_w]
    if plot_df.empty:
        st.info("No securities match current filters."); return

    # render
    fig = _scatter_quadrant(
        plot_df, x_col, y_col,
        None if color=="None" else color.lower(),
        size_mode="abs_net_weight",
        title=f"{y_col} vs {x_col}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # data preview
    with st.expander("Underlying Data"):
        base_cols = ["Security Name","Sector","Category","Net_weight", x_col, y_col]
        extra     = [c for c in numeric if c not in base_cols]
        st.dataframe(
            plot_df[base_cols+extra].sort_values(by=y_col, ascending=False),
            use_container_width=True
        )

    st.caption(f"{book} | {window} | rows={base.shape[0]} | tickers={base.shape[1]}")