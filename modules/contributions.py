import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import riskfolio as rf
import cvxpy as cp
from io import StringIO

# ------------------ Look-back map ------------------ #
_LOOKBACK_DAYS = {"6M": 126, "1Y": 252, "3Y": 756}

# ------------------ Core Computation (cached) ------------------ #
@st.cache_data(show_spinner="Computing risk contributions...")
def compute_risk_contributions_windowed(
    returns_full: pd.DataFrame,
    weights: pd.Series,
    rm: str,
    alpha: float,
    window_label: str
) -> tuple[pd.DataFrame, int]:
    """
    Returns per_asset df with BOTH signed and absolute %RC:
      - RC_pct_signed : sums to 100% (positives + negatives)
      - RC_pct_abs    : sums to 100% (all positive)
    """
    R = returns_full.copy()

    # Ensure chronological order if 'date' present
    if "date" in R.columns:
        R["date"] = pd.to_datetime(R["date"])
        R = R.sort_values("date")

    days = _LOOKBACK_DAYS[window_label]
    R = R.iloc[-days:].copy()

    if "date" in R.columns:
        R = R.drop(columns="date")

    tickers = weights.dropna().index
    R = R.reindex(columns=tickers)

    # Clean NA
    R = R.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if R.isna().any().any():
        R = R.fillna(method="ffill").fillna(method="bfill")
        R = R.dropna(axis=0, how="any")

    # Drop constants
    const_cols = [c for c in R.columns if R[c].nunique() <= 1]
    if const_cols:
        R = R.drop(columns=const_cols)
        weights = weights.drop(const_cols, errors="ignore")

    if R.empty:
        raise ValueError("No return observations after cleaning.")
    if R.shape[0] < 5:
        raise ValueError(f"Insufficient rows ({R.shape[0]}) for stable RC (window={window_label}).")

    # Scale if in %
    if R.values.max() > 2:
        R = R / 100.0

    # Align weights, normalize by gross so sign preserved
    w = weights.reindex(R.columns).fillna(0.0)
    if w.abs().sum() == 0:
        raise ValueError("All aligned weights are zero.")
    w = w / w.abs().sum()

    eff_alpha = alpha
    if rm.upper() in ("CVAR", "EVAR", "CDAR") and R.shape[0] * alpha < 1:
        eff_alpha = max(alpha, 1.0 / max(R.shape[0], 2))

    cov = R.cov()

    installed = cp.installed_solvers()
    if not installed:
        raise ValueError("No CVXPY solvers installed.")
    for pref in ["CLARABEL", "ECOS", "OSQP", "SCS"]:
        if pref in installed:
            solver = pref
            break
    else:
        solver = installed[0]

    rc_raw_arr = rf.Risk_Contribution(
        w       = w.to_frame(name="w"),
        returns = R,
        cov     = cov,
        rm      = rm,
        alpha   = eff_alpha,
        solver  = solver
    ).flatten()

    if rc_raw_arr.size == 0:
        raise ValueError("Risk_Contribution produced empty result.")

    rc_raw = pd.Series(rc_raw_arr, index=w.index, name="RC_raw")

    # Signed %
    signed_total = rc_raw.sum()
    rc_pct_signed = rc_raw * 0 if np.isclose(signed_total, 0) else rc_raw / signed_total * 100.0

    # Absolute %
    rc_abs = rc_raw.abs()
    abs_total = rc_abs.sum()
    rc_pct_abs = rc_abs * 0 if np.isclose(abs_total, 0) else rc_abs / abs_total * 100.0

    per_asset = (pd.DataFrame({
                    "Ticker": w.index,
                    "Weight": w.values,
                    "RC_raw": rc_raw.values,
                    "RC_pct_signed": rc_pct_signed.values,
                    "RC_pct_abs": rc_pct_abs.values
                 })
                 .assign(Position=lambda d: np.where(d.Weight >= 0, "Long", "Short"))
                 .reset_index(drop=True))

    return per_asset, R.shape[0]

# ------------------ Aggregations ------------------ #
def aggregate_by_sector(per_asset: pd.DataFrame, portfolio_df: pd.DataFrame, col: str) -> pd.DataFrame:
    if 'EOD Ticker' not in portfolio_df.columns or 'Sector' not in portfolio_df.columns:
        raise ValueError("portfolio_df must contain 'EOD Ticker' and 'Sector'.")
    sector_map = portfolio_df.set_index('EOD Ticker')['Sector']
    df = per_asset.copy()
    df["Sector"] = df["Ticker"].map(sector_map)
    agg = (df.groupby("Sector", dropna=False)[col]
             .sum()
             .sort_values(ascending=False)
             .reset_index())
    return agg

def sector_long_short_breakdown(per_asset: pd.DataFrame, portfolio_df: pd.DataFrame, col: str) -> pd.DataFrame:
    if 'EOD Ticker' not in portfolio_df.columns or 'Sector' not in portfolio_df.columns:
        raise ValueError("portfolio_df must contain 'EOD Ticker' and 'Sector'.")
    sector_map = portfolio_df.set_index('EOD Ticker')['Sector']
    df = per_asset.copy()
    df["Sector"] = df["Ticker"].map(sector_map)
    stacked = (df.groupby(["Sector","Position"], dropna=False)[col]
                 .sum()
                 .reset_index())
    # order sectors by total value desc
    order = (stacked.groupby("Sector")[col]
                    .sum()
                    .sort_values(ascending=False)
                    .index.tolist())
    stacked["Sector"] = pd.Categorical(stacked["Sector"], categories=order, ordered=True)
    stacked = stacked.sort_values(["Sector","Position"])
    return stacked

# ------------------ Plotting Helpers ------------------ #
def plot_per_asset(df: pd.DataFrame, rm: str, alpha: float, window: str, value_col: str):
    title_suffix = "Absolute" if value_col.endswith("_abs") else "Signed"
    plot_df = df.sort_values(value_col, ascending=False)
    order = plot_df["Ticker"].tolist()

    fig = px.bar(
        plot_df,
        x="Ticker",
        y=value_col,
        color="Position",
        color_discrete_map={"Long":"#00FF00","Short":"#FF4136"},
        title=f"{rm} Risk Contribution ({title_suffix}, α={alpha:.0%}, {window})",
        labels={value_col:"% of Total Risk" + (" (Abs)" if title_suffix == "Absolute" else "")},
        hover_data={"Weight":":.4f","RC_raw":":.4f","RC_pct_signed":":.2f","RC_pct_abs":":.2f"},
        template="plotly_dark",
        category_orders={"Ticker": order}
    )
    fig.update_layout(
        xaxis_tickangle=-90,
        xaxis_tickfont=dict(size=9, color="white"),
        margin=dict(b=110, t=60),
        legend_title_text=""
    )
    fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.8)
    return fig

def plot_sector(df_sector: pd.DataFrame, rm: str, alpha: float, window: str, value_col: str):
    title_suffix = "Absolute" if value_col.endswith("_abs") else "Signed"
    df_sector = df_sector.sort_values(value_col, ascending=False)
    order = df_sector["Sector"].tolist()

    fig = px.bar(
        df_sector,
        x="Sector",
        y=value_col,
        title=f"{rm} Sector Contribution ({title_suffix}, α={alpha:.0%}, {window})",
        labels={value_col:"% of Total Risk" + (" (Abs)" if title_suffix == "Absolute" else "")},
        template="plotly_dark",
        category_orders={"Sector": order}
    )
    fig.update_layout(margin=dict(t=60, b=80))
    fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.8)
    return fig

def plot_sector_stacked_long_short(stacked_df: pd.DataFrame, rm: str, alpha: float, window: str, value_col: str):
    # sectors already categorical & sorted in helper
    fig = px.bar(
        stacked_df,
        x="Sector",
        y=value_col,
        color="Position",
        color_discrete_map={"Long":"#00FF00","Short":"#FF4136"},
        title=f"{rm} Sector Long/Short Breakdown (Signed, α={alpha:.0%}, {window})",
        labels={value_col:"% of Total Risk"},
        template="plotly_dark"
    )
    fig.update_layout(barmode="relative", margin=dict(t=60, b=80))
    fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.8)
    return fig

# ------------------ Utilities ------------------ #
def _get_net_weights(portfolio_df: pd.DataFrame) -> pd.Series:
    if 'EOD Ticker' not in portfolio_df.columns or 'Net_weight' not in portfolio_df.columns:
        raise ValueError("portfolio_df must include 'EOD Ticker' and 'Net_weight'.")
    return portfolio_df.set_index('EOD Ticker')['Net_weight'].fillna(0.0)

def _to_csv_download(df: pd.DataFrame) -> bytes:
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ------------------ Page Renderer ------------------ #
def render_contributions(master_df, longs_df, shorts_df, portfolio_df, page_header):
    """
    Windowed contributions using master_df + Net_weight.
    """
    page_header("", "Contributions",
                "Per-asset & sector % risk contribution (windowed)")

    with st.sidebar:
        st.markdown("**Contributions Controls**")
        window_choice = st.multiselect("Look-back", ["6M","1Y","3Y"], default=["1Y"], max_selections=1)
        rm_options = ["CVaR", "CDaR", "EVaR", "MAD"]
        rm_sel = st.multiselect("Risk Measure", rm_options, default=["CVaR"], max_selections=1)
        alpha = st.slider("Alpha (tail probability)", 0.01, 0.10, 0.05, 0.01)
        # Default = Absolute
        absolute_mode = st.toggle("Show Absolute Contributions", value=True)
        show_sector = st.toggle("Show Sector Aggregation", value=True)
        show_sector_stacked = st.toggle(
            "Stack Sector Long/Short",
            value=False,
            disabled=not show_sector or absolute_mode
        )
        show_raw_head = st.toggle("Show Raw Asset Table Head", value=False)

    if not window_choice or not rm_sel:
        st.info("Select Look-back and Risk Measure.")
        return

    window = window_choice[0]
    rm = rm_sel[0]

    if master_df.empty:
        st.warning("No return data available.")
        return

    try:
        weights = _get_net_weights(portfolio_df)
    except Exception as e:
        st.error(f"Weight selection error: {e}")
        return

    # Align return columns first
    full = master_df[[c for c in master_df.columns if c in weights.index]]
    if full.empty:
        st.warning("No overlapping tickers between returns and portfolio.")
        return

    # Compute per-asset contributions
    try:
        per_asset, eff_rows = compute_risk_contributions_windowed(
            returns_full=full,
            weights=weights,
            rm=rm,
            alpha=alpha,
            window_label=window
        )
    except ValueError as e:
        st.warning(str(e))
        return
    except Exception as e:
        st.error(f"Contribution computation failed: {e}")
        return

    # Column to show
    value_col = "RC_pct_abs" if absolute_mode else "RC_pct_signed"

    # Per-asset plot
    st.subheader("Per-Asset Contributions")
    fig_asset = plot_per_asset(per_asset, rm, alpha, window, value_col)
    st.plotly_chart(fig_asset, use_container_width=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download Asset Contributions CSV",
            data=_to_csv_download(per_asset),
            file_name=f"asset_contributions_{rm}_{window}.csv",
            mime="text/csv"
        )

    if show_raw_head:
        with st.expander("Raw Asset Contribution Table"):
            st.dataframe(per_asset, use_container_width=True)

    # Sector aggregation
    if show_sector:
        st.subheader("Sector Contributions")
        try:
            sector_df = aggregate_by_sector(per_asset, portfolio_df, value_col)
            fig_sector = plot_sector(sector_df, rm, alpha, window, value_col)
            st.plotly_chart(fig_sector, use_container_width=True)

            if show_sector_stacked:  # only enabled when signed
                try:
                    stacked_df = sector_long_short_breakdown(per_asset, portfolio_df, value_col)
                    fig_stacked = plot_sector_stacked_long_short(stacked_df, rm, alpha, window, value_col)
                    st.plotly_chart(fig_stacked, use_container_width=True)
                except Exception as e:
                    st.warning(f"Sector long/short breakdown unavailable: {e}")

            with col_dl2:
                st.download_button(
                    "Download Sector Contributions CSV",
                    data=_to_csv_download(sector_df),
                    file_name=f"sector_contributions_{rm}_{window}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.warning(f"Sector aggregation unavailable: {e}")

    st.caption(
        f"{window} | rm={rm} | alpha={alpha:.2f} | rows={eff_rows} | "
        f"tickers={per_asset.shape[0]} | mode={'Abs' if absolute_mode else 'Signed'} | "
        f"signed_sum={per_asset['RC_pct_signed'].sum():.2f}% | abs_sum={per_asset['RC_pct_abs'].sum():.2f}%"
    )