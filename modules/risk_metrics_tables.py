import numpy as np
import pandas as pd
import streamlit as st
import riskfolio as rf   # direct riskfolio v7.0.1

# --- Small compatibility helpers ---
def SemiDeviation(x):
    return rf.SemiDeviation(np.asarray(x).ravel())

def WR(x):
    x = np.asarray(x).ravel()
    return (x > 0).mean()

# Safe‑divide helpers
def _safe_divide(numer, denom, default=0.0):
    try:
        return numer / denom
    except (ZeroDivisionError, FloatingPointError):
        return default

def _safe_divide_array(numer, denom):
    numer_arr = np.asarray(numer, dtype=float)
    denom_arr = np.asarray(denom, dtype=float)
    # where denom is zero, yield 0.0
    return np.where(denom_arr == 0.0, 0.0, numer_arr / denom_arr)

_WINDOW_DAYS = {'1M':21, '3M':63, '6M':126, '1Y':252}  # no “FullHist” here

def _geom_wealth(r):
    return np.cumprod(1 + r.flatten())

def _geom_drawdowns(r):
    W = _geom_wealth(r)
    peak = np.maximum.accumulate(W)
    # guard against peak == W == 0
    return _safe_divide_array((peak - W), peak)

def compute_metrics_window(sub, alpha=0.05):
    recs = {}
    for t in sub.columns:
        x = sub[t].dropna().values.reshape(-1,1)
        if x.shape[0] < 2:
            continue

        evar, _ = rf.EVaR_Hist(x, alpha=alpha)
        recs[t] = {
            'MAD':     rf.MAD(x),
            'SemiDev': SemiDeviation(x),
            'VaR95':   rf.VaR_Hist(x, alpha=alpha),
            'CVaR95':  rf.CVaR_Hist(x, alpha=alpha),
            'EVaR95':  evar,
            'RLVaR95': rf.RLVaR_Hist(x, alpha=alpha),
            'TG95':    rf.TG(x, alpha=alpha),
            'WR':      WR(x),
            'LPM1':    rf.LPM(x, MAR=0, p=1),
            'LPM2':    rf.LPM(x, MAR=0, p=2)
        }
    return pd.DataFrame(recs).T

def compute_metrics_window_geom(sub, alpha=0.05):
    recs = {}
    for t in sub.columns:
        x = sub[t].dropna().values.reshape(-1,1)
        if x.shape[0] < 2:
            continue

        dd = _geom_drawdowns(x)
        if dd.size == 0:
            continue

        # Max drawdown
        mdd = float(dd.max())

        # Average of only positive drawdowns
        positive_dd = dd[dd > 0]
        add = float(positive_dd.mean()) if positive_dd.size > 0 else 0.0

        # 95th‑percentile drawdown (DaR)
        dar = float(np.percentile(dd, 95)) if dd.size > 0 else 0.0

        # Conditional DaR
        tail = dd[dd >= dar]
        cdar = float(tail.mean()) if tail.size > 0 else dar

        # Ulcer index = sqrt(mean(square(dd)))
        uci = float(np.sqrt((dd**2).mean())) if dd.size > 0 else 0.0

        recs[t] = {
            'MDD_Abs':   mdd,
            'ADD_Abs':   add,
            'DaR_Abs95': dar,
            'CDaR_Abs95': cdar,
            'UCI_Abs':   uci
        }

    return pd.DataFrame(recs).T

@st.cache_data(show_spinner="Computing risk metric tables…")
def build_and_split(returns_df, returns_label, portData, window_label, alpha=0.05):
    df = returns_df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    wcol = {'FullHist':'Net_weight','Long':'Long_weight','Short':'Short_weight'}[returns_label]
    meta = portData.set_index('EOD Ticker')

    days = _WINDOW_DAYS[window_label]
    if days:
        df = df.iloc[-days:]

    tickers = [t for t in df.columns if t in meta.index]
    sub = df[tickers]

    desc_full = compute_metrics_window(sub, alpha=alpha)
    tail      = desc_full[['VaR95','CVaR95','EVaR95','RLVaR95','TG95','WR','LPM1','LPM2']]
    desc      = desc_full[['MAD','SemiDev']]
    dd        = compute_metrics_window_geom(sub, alpha=alpha)

    def attach(tbl, pct_cols, dec_cols=[]):
        tbl = tbl.loc[tbl.index.intersection(meta.index)].copy()
        tbl.insert(0, 'Security Name', meta.loc[tbl.index, 'EOD Name'])
        w = meta[wcol].reindex(tbl.index).fillna(0)
        # weight % safely
        weights = _safe_divide_array(w, 1.0) * 100.0
        tbl.insert(1, 'Weight', weights.round(2).astype(str) + '%')

        for c in pct_cols:
            if c in tbl.columns:
                tbl[c] = (tbl[c] * 100).round(2).astype(str) + '%'
        for c in dec_cols:
            if c in tbl.columns:
                tbl[c] = tbl[c].round(4)
        return tbl

    return {
        'descriptive': attach(desc, ['MAD','SemiDev']),
        'tail_risk':   attach(tail, tail.columns.tolist()),
        'drawdowns':   attach(dd,   dd.columns.tolist())
    }

def render(master_df, longs_df, shorts_df, portfolio_df, page_header):
    page_header("","Risk Metric Tables","Descriptive, tail-risk & drawdown metrics")

    with st.sidebar:
        st.markdown("**Risk Metric Controls**")
        returns_label = st.multiselect("Book", ["FullHist","Long","Short"],
                                       default=["FullHist"], max_selections=1)
        window_label  = st.multiselect("Look-back", list(_WINDOW_DAYS.keys()),
                                       default=["1Y"], max_selections=1)

    if not returns_label or not window_label:
        st.info("Select a book and a look-back window.")
        return

    returns_label = returns_label[0]
    window_label  = window_label[0]
    base = {'FullHist':master_df, 'Long':longs_df, 'Short':shorts_df}[returns_label]

    try:
        tables = build_and_split(base, returns_label, portfolio_df, window_label)
    except Exception as e:
        st.error(f"Metric computation failed: {e}")
        return

    tabs = st.tabs(["Descriptive","Tail Risk","Drawdowns"])
    tabs[0].dataframe(tables['descriptive'], use_container_width=True)
    tabs[1].dataframe(tables['tail_risk'],   use_container_width=True)
    tabs[2].dataframe(tables['drawdowns'],   use_container_width=True)

    st.caption(f"{returns_label} | {window_label} | rows={base.shape[0]} tickers={base.shape[1]}")
    st.session_state['risk_tables_latest'] = tables
    st.session_state['risk_portfolio_df'] = portfolio_df