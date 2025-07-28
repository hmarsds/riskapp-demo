import re
import inspect

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import riskfolio as rf

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _geom_drawdowns(r):
    W    = np.cumprod(1 + r.flatten())
    peak = np.maximum.accumulate(W)
    return (peak - W) / peak

# Only 6M (126) and 1Y (252) windows
_WINDOW_DAYS = {'6M': 126, '1Y': 252}

# Geometric drawdown metrics
_GEOM_METRICS = {
    'MDD_Abs', 'ADD_Abs', 'DaR_Abs', 'CDaR95_Abs', 'EDaR_Abs', 'UCI_Abs'
}

# User‐friendly → RF function
_METRIC_ALIASES = {
    'cvar95':     'CVaR_Hist',
    'mdd_abs':    'MDD_Abs',
    'add_abs':    'ADD_Abs',
    'dar_abs':    'DaR_Abs',
    'cdar95_abs': 'CDaR95_Abs',
}

# Metrics that accept alpha
_ALPHA_METRICS = {'CVaR_Hist', 'DaR_Abs', 'CDaR95_Abs', 'EDaR_Abs'}


@st.cache_data(show_spinner="Rendering heatmap…")
def _build_heatmap_figure(
    returns_df: pd.DataFrame,
    display_metric: str,
    alpha: float,
    window_label: str,
    freq: str,
    colorscale: str
) -> go.Figure:
    """Produce a Plotly heatmap of rolling risk metrics per asset."""
    # alias the metric
    clean = re.sub(r'[^A-Za-z0-9]', '', display_metric).lower()
    metric_key = _METRIC_ALIASES.get(clean, display_metric)

    if (metric_key not in _GEOM_METRICS
            and not hasattr(rf.RiskFunctions, metric_key)):
        raise ValueError(f"Unrecognized metric '{display_metric}'.")
    use_geom = metric_key in _GEOM_METRICS

    # grab the RF function for non‑geom
    func = None
    needs_alpha = False
    if not use_geom:
        func = getattr(rf.RiskFunctions, metric_key)
        sig = inspect.signature(func)
        needs_alpha = 'alpha' in sig.parameters

    # ensure a DatetimeIndex
    df = returns_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ('date', 'Date'):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                break
        else:
            raise ValueError("Need a DatetimeIndex or a 'date' column")
    df.sort_index(inplace=True)

    # build rolling endpoints
    window = _WINDOW_DAYS[window_label]
    raw_ends = df.resample(freq).last().index
    idx = df.index
    endpoints = []
    for dt in raw_ends:
        pos = idx.searchsorted(dt, 'right') - 1
        if pos >= (window - 1):
            endpoints.append(idx[pos])

    tickers = df.columns.tolist()
    Z = np.full((len(tickers), len(endpoints)), np.nan, dtype=float)

    # fill the matrix
    for j, end in enumerate(endpoints):
        slice_ = df.loc[:end].tail(window)
        for i, t in enumerate(tickers):
            arr = slice_[t].dropna().values
            if arr.size < 2:
                continue

            if use_geom:
                dd = _geom_drawdowns(arr.reshape(-1, 1))
                if metric_key == 'MDD_Abs':
                    Z[i, j] = dd.max()
                elif metric_key == 'ADD_Abs':
                    Z[i, j] = dd[dd > 0].mean() if np.any(dd > 0) else 0.0
                elif metric_key == 'DaR_Abs':
                    Z[i, j] = np.percentile(dd, 95)
                elif metric_key == 'CDaR95_Abs':
                    dar = np.percentile(dd, 95)
                    tail = dd[dd >= dar]
                    Z[i, j] = tail.mean() if len(tail) else dar
                elif metric_key == 'EDaR_Abs':
                    Z[i, j] = (1 / alpha) * np.log(np.mean(np.exp(alpha * dd)))
                elif metric_key == 'UCI_Abs':
                    Z[i, j] = np.sqrt(np.mean(dd ** 2))
            else:
                val = func(arr if not needs_alpha else arr, alpha=alpha)
                Z[i, j] = val * 100

    # scale geom‐metrics back to percent
    z_vals = Z * 100 if use_geom else Z

    # If CVaR95, cap the color domain at its 5th/95th percentiles
    zmin = zmax = None
    if display_metric == "CVaR95":
        flat = z_vals.flatten()
        flat = flat[~np.isnan(flat)]
        if flat.size:
            zmin, zmax = np.percentile(flat, [5, 95])

    # build the heatmap trace
    trace = go.Heatmap(
        z=z_vals,
        x=[d.strftime("%Y-%m") for d in endpoints],
        y=tickers,
        colorscale=colorscale,
        colorbar=dict(title=f"{display_metric} (%)", tickformat=".1f", len=0.7),
        hovertemplate=f"<b>%{{y}}</b><br>%{{x}}<br>{display_metric}: %{{z:.2f}}%<extra></extra>",
        zmin=zmin,
        zmax=zmax
    )

    fig = go.Figure(trace)
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"Rolling {window_label} {display_metric} (%)"
                 + (f" @α={alpha:.2f}" if metric_key in _ALPHA_METRICS else ""),
            x=0.5
        ),
        xaxis=dict(tickangle=-45, showgrid=False),
        yaxis=dict(
            autorange='reversed',
            showgrid=False,
            tickmode='array',
            tickvals=tickers
        ),
        margin=dict(l=100, r=50, t=100, b=120),
        height=1200
    )

    return fig


def render(returns_df: pd.DataFrame, page_header):
    # Header
    page_header("", "Heatmap", "Rolling risk metrics per asset")

    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Heatmap Options")

    # Only these five metrics, in order
    METRICS = [
        "CVaR95",
        "MDD_Abs",
        "ADD_Abs",
        "DaR_Abs",
        "CDaR95_Abs"
    ]
    display_metric = st.sidebar.selectbox("Metric", METRICS, index=0)

    # Alpha slider now 0.01–0.10
    alpha = st.sidebar.slider(
        "Alpha (for CVaR95…)",
        min_value=0.01, max_value=0.10,
        value=0.05, step=0.01
    )

    # Only 6M or 1Y window
    WINDOWS = ["6M", "1Y"]
    window_label = st.sidebar.selectbox("Window", WINDOWS, index=0)

    # Frequency fixed to monthly
    freq = "M"

    # Colorscale choices
    COLORSCALES = ["plasma", "viridis", "cividis", "inferno"]
    colorscale = st.sidebar.selectbox("Colorscale", COLORSCALES, index=0)

    # Build & render
    fig = _build_heatmap_figure(
        returns_df,
        display_metric,
        alpha,
        window_label,
        freq,
        colorscale
    )
    st.plotly_chart(fig, use_container_width=True)