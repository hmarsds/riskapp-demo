import numpy as np
import pandas as pd
import streamlit as st
import riskfolio as rf   # v7.0.1

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

_WINDOW_DAYS = {'1M':21,'3M':63,'6M':126,'1Y':252}

def _geom_wealth(r):
    return np.cumprod(1.0 + r.flatten()).astype(float)

def _geom_drawdowns(r):
    W = _geom_wealth(r)
    peak = np.maximum.accumulate(W)
    dd = ((peak - W) / peak).astype(float)
    st.write("  [geom_drawdowns] sample:", dd[:5].flatten())  # inline log
    return dd

# ——— Core metrics ———
def compute_metrics_window(sub, alpha=0.05):
    recs = {}
    for t in sub.columns:
        x = sub[t].dropna().values.reshape(-1,1).astype(float)
        st.write(f"[compute_metrics_window] {t} dtype:", x.dtype, "shape:", x.shape)
        if len(x) < 2:
            continue
        evar,_ = rf.EVaR_Hist(x, alpha=alpha)
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
            'LPM2':    float(rf.LPM(x, MAR=0, p=2)),
        }
    df = pd.DataFrame(recs).T
    st.write("[compute_metrics_window] head:", df.head())
    return df

def compute_metrics_window_geom(sub, alpha=0.05):
    recs = {}
    for t in sub.columns:
        x = sub[t].dropna().values.reshape(-1,1).astype(float)
        st.write(f"[compute_metrics_window_geom] {t} shape:", x.shape)
        if len(x) < 2:
            continue
        dd = _geom_drawdowns(x)
        mdd = float(dd.max())
        add = float(dd[dd>0].mean()) if np.any(dd>0) else 0.0
        dar = float(np.percentile(dd,95))
        tail = dd[dd>=dar]
        cdar = float(tail.mean()) if len(tail) else dar
        uci  = float(np.sqrt(np.mean(dd**2)))
        recs[t] = {
            'MDD_Abs':   mdd,
            'ADD_Abs':   add,
            'DaR_Abs95': dar,
            'CDaR_Abs95':cdar,
            'UCI_Abs':   uci
        }
    df = pd.DataFrame(recs).T
    st.write("[compute_metrics_window_geom] head:", df.head())
    return df

# ——— Build & split (NO cache during debugging) ———
def build_and_split(returns_df, returns_label, portData, window_label, alpha=0.05):
    st.write(f"--- build_and_split: {returns_label=}  {window_label=} ---")
    df = returns_df.copy()

    # date → index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # weight lookup
    wcol = {'FullHist':'Net_weight','Long':'Long_weight','Short':'Short_weight'}[returns_label]
    st.write("using weight column:", wcol)

    meta = portData.set_index('EOD Ticker')
    days = _WINDOW_DAYS[window_label]
    if days:
        df = df.iloc[-days:]
    st.write("post-slice returns shape:", df.shape)

    # only numeric
    sub = df[[c for c in df.columns if c in meta.index]]
    sub = sub.apply(pd.to_numeric, errors='coerce')
    st.write("after to_numeric, dtypes:", sub.dtypes)

    desc_full = compute_metrics_window(sub, alpha=alpha)
    tail      = desc_full[['VaR95','CVaR95','EVaR95','RLVaR95','TG95','WR','LPM1','LPM2']]
    desc      = desc_full[['MAD','SemiDev']]
    dd        = compute_metrics_window_geom(sub, alpha=alpha)

    def attach(tbl, pct_cols, dec_cols=[]):
        tbl = tbl.loc[tbl.index.intersection(meta.index)].copy()
        tbl.insert(0, 'Security Name', meta.loc[tbl.index, 'EOD Name'])

        # 1) compute raw weights as floats
        w_series = meta[wcol].reindex(tbl.index).fillna(0).astype(float)
        weights = (w_series.values / 1.0) * 100.0  # numpy array of floats
        weights = np.round(weights, 2)

        # 2) turn into list of "xx.xx%" strings
        weight_strs = [f"{w:.2f}%" for w in weights]
        tbl.insert(1, 'Weight', weight_strs)

        # 3) for each pct column, build strings too
        for c in pct_cols:
            if c in tbl.columns:
                # tbl[c] is currently float like 0.1234 → we want "12.34%"
                vals = (tbl[c].astype(float) * 100.0).round(2)
                tbl[c] = [f"{v:.2f}%" for v in vals]

        # 4) decimals
        for c in dec_cols:
            if c in tbl.columns:
                tbl[c] = tbl[c].astype(float).round(4)

        return tbl

    return {
        'descriptive': attach(desc,   ['MAD','SemiDev']),
        'tail_risk':   attach(tail,   tail.columns.tolist()),
        'drawdowns':   attach(dd,     dd.columns.tolist()),
    }

# ——— Renderer ———
def render(master_df, longs_df, shorts_df, portfolio_df, page_header):
    page_header("", "Risk Metric Tables", "…")
    with st.sidebar:
        returns_label = st.multiselect("Book", ["FullHist","Long","Short"], default=["FullHist"], max_selections=1)
        window_label  = st.multiselect("Look‑back", list(_WINDOW_DAYS.keys()), default=["1Y"], max_selections=1)
    if not returns_label or not window_label:
        st.info("Pick book & window"); return
    tables = build_and_split(master_df, returns_label[0], portfolio_df, window_label[0])
    tabs = st.tabs(["Descriptive","Tail Risk","Drawdowns"])
    tabs[0].dataframe(tables['descriptive'], use_container_width=True)
    tabs[1].dataframe(tables['tail_risk'],   use_container_width=True)
    tabs[2].dataframe(tables['drawdowns'],   use_container_width=True)
    st.caption(f"{returns_label[0]} | {window_label[0]} | rows={master_df.shape[0]} tickers={master_df.shape[1]}")
    st.session_state['risk_tables_latest'] = tables
    st.session_state['risk_portfolio_df']   = portfolio_df