import numpy as np
import pandas as pd
import streamlit as st
import riskfolio as rf   # direct Riskfolio‑Lib v7.0.1

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

# look‑back windows
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
    for ticker in sub.columns:
        x = sub[ticker].dropna().values.reshape(-1,1).astype(float)
        if len(x) < 2:
            continue

        # guard EVaR against z=0
        try:
            evar, _ = rf.EVaR_Hist(x, alpha=alpha)
        except ZeroDivisionError:
            evar = np.nan

        recs[ticker] = {
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
    for ticker in sub.columns:
        x = sub[ticker].dropna().values.reshape(-1,1).astype(float)
        if len(x) < 2:
            continue

        dd = _geom_drawdowns(x)
        mdd  = float(dd.max())
        add  = float(dd[dd>0].mean()) if np.any(dd>0) else 0.0
        dar  = float(np.percentile(dd, 95))
        tail = dd[dd>=dar]
        cdar = float(tail.mean()) if len(tail) else dar
        uci  = float(np.sqrt(np.mean(dd**2)))

        recs[ticker] = {
            'MDD_Abs':    mdd,
            'ADD_Abs':    add,
            'DaR_Abs95':  dar,
            'CDaR_Abs95': cdar,
            'UCI_Abs':    uci
        }
    return pd.DataFrame(recs).T

# ——— Build & split wrapper ———
@st.cache_data(show_spinner="Computing risk metric tables…")
def build_and_split(returns_df, returns_label, portData, window_label, alpha=0.05):
    df = returns_df.copy()

    # convert date column to index if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # pick correct weight column
    wcol = {'FullHist':'Net_weight', 'Long':'Long_weight', 'Short':'Short_weight'}[returns_label]
    meta = portData.set_index('EOD Ticker')

    # slice lookback window
    days = _WINDOW_DAYS[window_label]
    if days:
        df = df.iloc[-days:]

    # restrict to tickers we have metadata for
    tickers = [t for t in df.columns if t in meta.index]
    sub = df[tickers].apply(pd.to_numeric, errors='coerce')

    # compute all three tables
    desc_full = compute_metrics_window(sub, alpha=alpha)
    tail      = desc_full[['VaR95','CVaR95','EVaR95','RLVaR95','TG95','WR','LPM1','LPM2']]
    desc      = desc_full[['MAD','SemiDev']]
    dd        = compute_metrics_window_geom(sub, alpha=alpha)

    def attach(tbl: pd.DataFrame, pct_cols: list[str], dec_cols: list[str] = []) -> pd.DataFrame:
        # limit to tickers present in metadata
        tbl = tbl.loc[tbl.index.intersection(meta.index)].copy()

        # compute raw weights
        w_series = meta[wcol].reindex(tbl.index).fillna(0).astype(float)

        # filter by book:
        #  - Long  => only positive weights
        #  - Short => only negative weights
        #  - All   => any non-zero weight
        if returns_label == "Long":
            mask = w_series > 0
        elif returns_label == "Short":
            mask = w_series < 0
        else:  # FullHist / All
            mask = w_series != 0

        tbl      = tbl.loc[mask]
        w_series = w_series.loc[mask]

        # insert Security Name
        tbl.insert(0, 'Security Name', meta.loc[tbl.index, 'EOD Name'])

        # format Weight column
        weight_strs = [f"{100*w:.2f}%" for w in w_series.values]
        tbl.insert(1, 'Weight', weight_strs)

        # format percent metrics
        for c in pct_cols:
            if c in tbl.columns:
                vals = (tbl[c].astype(float) * 100.0).round(2)
                tbl[c] = [f"{v:.2f}%" for v in vals]

        # format decimal metrics
        for c in dec_cols:
            if c in tbl.columns:
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
        window_label  = st.multiselect("Look‑back", list(_WINDOW_DAYS.keys()),
                                       default=["1Y"], max_selections=1)

    if not returns_label or not window_label:
        st.info("Select a book and a look‑back window.")
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
    st.session_state['risk_tables_latest']  = tables
    st.session_state['risk_portfolio_df']   = portfolio_df