import numpy as np
import pandas as pd
import streamlit as st
import riskfolio as rf  # v7.0.1

# ——— Helpers ———
def SemiDeviation(x):
    return rf.SemiDeviation(np.asarray(x, float).ravel())

def WR(x):
    arr = np.asarray(x, float).ravel()
    return float((arr > 0).mean())

def _safe_divide_array(numer, denom):
    na = np.asarray(numer, float)
    da = np.asarray(denom,  float)
    return np.where(da == 0.0, 0.0, na / da)

_WINDOW_DAYS = {'1M': 21, '3M': 63, '6M': 126, '1Y': 252}

def _geom_wealth(r):
    return np.cumprod(1.0 + r.flatten()).astype(float)

def _geom_drawdowns(r):
    W    = _geom_wealth(r)
    peak = np.maximum.accumulate(W)
    return ((peak - W) / peak).astype(float)

# ——— Core metrics ———
def compute_metrics_window(sub: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    recs = {}
    for ticker in sub.columns:
        x = sub[ticker].dropna().values.reshape(-1, 1).astype(float)
        if len(x) < 2:
            continue

        try:
            evar, _ = rf.EVaR_Hist(x, alpha=alpha)
        except ZeroDivisionError:
            evar = np.nan

        try:
            mad = rf.MAD(x)
        except Exception:
            mad = np.nan

        recs[ticker] = {
            'MAD':     float(mad),
            'SemiDev': float(SemiDeviation(x)),
            'VaR95':   float(rf.VaR_Hist(x, alpha=alpha)),
            'CVaR95':  float(rf.CVaR_Hist(x, alpha=alpha)),
            'EVaR95':  float(evar),
            'RLVaR95': float(rf.RLVaR_Hist(x, alpha=alpha)),
            'TG95':    float(rf.TG(x, alpha=alpha)),
            'WR':      float(WR(x)),
            'LPM1':    float(rf.LPM(x, MAR=0, p=1)),
            'LPM2':    float(rf.LPM(x, MAR=0, p=2)),
        }
    return pd.DataFrame(recs).T

def compute_metrics_window_geom(sub: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    recs = {}
    for ticker in sub.columns:
        x = sub[ticker].dropna().values.reshape(-1, 1).astype(float)
        if len(x) < 2:
            continue

        dd = _geom_drawdowns(x)
        mdd  = float(dd.max())
        add  = float(dd[dd > 0].mean()) if np.any(dd > 0) else 0.0
        dar  = float(np.percentile(dd, 95))
        tail = dd[dd >= dar]
        cdar = float(tail.mean()) if len(tail) else dar
        uci  = float(np.sqrt(np.mean(dd**2)))

        recs[ticker] = {
            'MDD_Abs':    mdd,
            'ADD_Abs':    add,
            'DaR_Abs95':  dar,
            'CDaR_Abs95': cdar,
            'UCI_Abs':    uci,
        }
    return pd.DataFrame(recs).T

# ——— Build & split ———
@st.cache_data(show_spinner="Computing risk metric tables…")
def build_and_split(
    returns_df: pd.DataFrame,
    returns_label: str,
    portData: pd.DataFrame,
    window_label: str,
    alpha: float = 0.05
) -> dict[str, pd.DataFrame]:
    df = returns_df.copy()

    # date → index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # pick weight column
    wcol = {'FullHist': 'Net_weight', 'Long': 'Long_weight', 'Short': 'Short_weight'}[returns_label]
    meta = portData.set_index('EOD Ticker')

    # window slice
    days = _WINDOW_DAYS.get(window_label)
    if days:
        df = df.iloc[-days:]

    # filter to tickers in metadata
    tickers = [t for t in df.columns if t in meta.index]
    sub = df[tickers].apply(pd.to_numeric, errors='coerce')

    # compute tables
    desc_full = compute_metrics_window(sub, alpha=alpha)
    tail      = desc_full[['VaR95','CVaR95','EVaR95','RLVaR95','TG95','WR','LPM1','LPM2']]
    desc      = desc_full[['MAD','SemiDev']]
    dd        = compute_metrics_window_geom(sub, alpha=alpha)

    def attach(tbl: pd.DataFrame, pct_cols: list[str], dec_cols: list[str] = []) -> pd.DataFrame:
        tbl = tbl.loc[tbl.index.intersection(meta.index)].copy()
        tbl.insert(0, 'Security Name', meta.loc[tbl.index, 'EOD Name'])

        # weights as floats → formatted strings
        w_series = meta[wcol].reindex(tbl.index).fillna(0).astype(float)
        weight_strs = [f"{100*w:.2f}%" for w in w_series.values]
        tbl.insert(1, 'Weight', weight_strs)

        # percent columns
        for c in pct_cols:
            if c in tbl:
                pct_vals = (tbl[c].astype(float) * 100).round(2)
                tbl[c] = [f"{v:.2f}%" for v in pct_vals]

        # decimal columns
        for c in dec_cols:
            if c in tbl:
                tbl[c] = tbl[c].astype(float).round(4)

        return tbl

    return {
        'descriptive': attach(desc,   ['MAD','SemiDev'], dec_cols=[]),
        'tail_risk':   attach(tail,   tail.columns.tolist(), dec_cols=[]),
        'drawdowns':   attach(dd,     dd.columns.tolist(), dec_cols=[]),
    }

# ——— Renderer ———
def render(
    master_df: pd.DataFrame,
    longs_df: pd.DataFrame,
    shorts_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    page_header
) -> None:
    page_header("", "Risk Metric Tables", "Descriptive, tail-risk & drawdown metrics")

    with st.sidebar:
        returns_label = st.multiselect("Book", ["FullHist","Long","Short"], default=["FullHist"], max_selections=1)
        window_label  = st.multiselect("Look-back", list(_WINDOW_DAYS.keys()), default=["1Y"], max_selections=1)

    if not returns_label or not window_label:
        st.info("Select a book and a look-back window.")
        return

    tables = build_and_split(master_df, returns_label[0], portfolio_df, window_label[0])
    tabs   = st.tabs(["Descriptive","Tail Risk","Drawdowns"])
    tabs[0].dataframe(tables['descriptive'], use_container_width=True)
    tabs[1].dataframe(tables['tail_risk'],    use_container_width=True)
    tabs[2].dataframe(tables['drawdowns'],    use_container_width=True)

    st.caption(f"{returns_label[0]} | {window_label[0]} | rows={master_df.shape[0]} tickers={master_df.shape[1]}")
    st.session_state['risk_tables_latest'] = tables
    st.session_state['risk_portfolio_df']   = portfolio_df