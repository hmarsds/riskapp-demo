# modules/risk_metrics_tables.py

import numpy as np
import pandas as pd
import streamlit as st
import riskfolio as rf

# ——— Helpers ———
def SemiDeviation(x):
    return rf.SemiDeviation(np.asarray(x, float).ravel())

def WR(x):
    arr = np.asarray(x, float).ravel()
    return float((arr > 0).mean())

def _safe_divide_array(numer, denom):
    na = np.asarray(numer, float)
    da = np.asarray(denom,  float)
    return np.where(da == 0.0, 0.0, na/da)

# look-back windows
_WINDOW_DAYS = {'1M':21, '3M':63, '6M':126, '1Y':252, 'FullHist':None}

def _geom_wealth(r):
    return np.cumprod(1.0 + r.flatten()).astype(float)

def _geom_drawdowns(r):
    W    = _geom_wealth(r)
    peak = np.maximum.accumulate(W)
    return ((peak - W) / peak).astype(float)

# ——— Core metrics ———
def compute_metrics_window(sub, alpha=0.05):
    recs = {}
    for t in sub.columns:
        x = sub[t].dropna().values.reshape(-1,1).astype(float)
        if len(x) < 2:
            continue

        try:
            evar, _ = rf.EVaR_Hist(x, alpha=alpha)
        except ZeroDivisionError:
            evar = np.nan

        recs[t] = {
            'MAD':     float(rf.MAD(x)),
            'SemiDev': float(SemiDeviation(x)),
            'VaR95':   float(rf.VaR_Hist(x, alpha=alpha)),
            'CVaR95':  float(rf.CVaR_Hist(x, alpha=alpha)),
            'EVaR95':  float(evar),
            'RLVaR95': float(rf.RLVaR_Hist(x, alpha=alpha)),
            'TG95':    float(rf.TG(x, alpha=alpha)),
            'WR':      float(WR(x)),
            'LPM1':    float(rf.LPM(x, MAR=0, p=1)),
            'LPM2':    float(rf.LPM(x, MAR=0, p=2))
        }
    return pd.DataFrame(recs).T

def compute_metrics_window_geom(sub, alpha=0.05):
    recs = {}
    for t in sub.columns:
        x = sub[t].dropna().values.reshape(-1,1).astype(float)
        if len(x) < 2:
            continue
        dd = _geom_drawdowns(x)
        recs[t] = {
            'MDD_Abs':    float(dd.max()),
            'ADD_Abs':    float(dd[dd>0].mean()) if np.any(dd>0) else 0.0,
            'DaR_Abs95':  float(np.percentile(dd, 95)),
            'CDaR_Abs95': float(np.mean(dd[dd>=np.percentile(dd, 95)])),
            'UCI_Abs':    float(np.sqrt(np.mean(dd**2)))
        }
    return pd.DataFrame(recs).T

# ——— Build & split ———
@st.cache_data(show_spinner="Computing risk metric tables…")
def build_and_split(returns_df, returns_label, portData, window_label, alpha=0.05):
    df = returns_df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    wcol = {'FullHist':'Net_weight', 'Long':'Long_weight', 'Short':'Short_weight'}[returns_label]
    meta = portData.set_index('EOD Ticker')

    days = _WINDOW_DAYS[window_label]
    if days:
        df = df.iloc[-days:]

    tickers = [t for t in df.columns if t in meta.index]
    sub = df[tickers].apply(pd.to_numeric, errors='coerce')

    desc_full = compute_metrics_window(sub, alpha=alpha)
    tail      = desc_full[['VaR95','CVaR95','EVaR95','RLVaR95','TG95','WR','LPM1','LPM2']]
    desc      = desc_full[['MAD','SemiDev']]
    dd        = compute_metrics_window_geom(sub, alpha=alpha)

    def attach(tbl, pct_cols, dec_cols=[]):
        tbl = tbl.loc[tbl.index.intersection(meta.index)].copy()
        w_series = meta[wcol].reindex(tbl.index).fillna(0).astype(float)

        # filter by book side
        if returns_label == "Long":
            mask = w_series > 0
        elif returns_label == "Short":
            mask = w_series < 0
        else:  # FullHist
            mask = w_series != 0

        tbl      = tbl.loc[mask]
        w_series = w_series.loc[mask]

        tbl.insert(0, 'Security Name', meta.loc[tbl.index, 'EOD Name'])
        weight_strs = [f"{100*w:.2f}%" for w in w_series.values]
        tbl.insert(1, 'Weight', weight_strs)

        for c in pct_cols:
            if c in tbl:
                vals = (tbl[c].astype(float)*100).round(2)
                tbl[c] = [f"{v:.2f}%" for v in vals]
        for c in dec_cols:
            if c in tbl:
                tbl[c] = tbl[c].astype(float).round(4)

        return tbl

    return {
        'descriptive': attach(desc,  ['MAD','SemiDev']),
        'tail_risk':   attach(tail,  tail.columns.tolist()),
        'drawdowns':   attach(dd,    dd.columns.tolist()),
    }

# ——— Renderer ———
def render(master_df, longs_df, shorts_df, portfolio_df, page_header):
    page_header("", "📊 Risk Metric Tables", "Descriptive, tail‑risk & drawdown metrics")

    with st.sidebar:
        returns_label = st.multiselect("Book", ["FullHist","Long","Short"],
                                       default=["FullHist"], max_selections=1)
        window_label  = st.multiselect("Look‑back", ["1M","3M","6M","1Y"],
                                       default=["1Y"], max_selections=1)

    if not returns_label or not window_label:
        st.info("Select a book and a window."); return

    rl = returns_label[0]
    wl = window_label[0]
    base = {'FullHist': master_df, 'Long': longs_df, 'Short': shorts_df}[rl]

    try:
        tables = build_and_split(base, rl, portfolio_df, wl)
    except Exception as e:
        st.error(f"Metric computation failed: {e}")
        return

    tabs = st.tabs(["Descriptive","Tail Risk","Drawdowns"])
    tabs[0].dataframe(tables['descriptive'], use_container_width=True)
    tabs[1].dataframe(tables['tail_risk'],   use_container_width=True)
    tabs[2].dataframe(tables['drawdowns'],   use_container_width=True)

    st.caption(f"{rl} | {wl} | rows={base.shape[0]} tickers={base.shape[1]}")
    st.session_state['risk_tables_latest'] = tables
    st.session_state['risk_portfolio_df']  = portfolio_df