import os
import streamlit as st
from utils.auth import require_access_code
from data_loader import load_data
from modules import (
    risk_metrics_tables,
    quadrants,
    contributions,
    heatmap,
    hierarchical,
    cluster_contributions,
    centrality,
    communities,
    #factor_analysis,
)

st.set_page_config(page_title="Investment Dashboard", page_icon="📊", layout="wide")

# ---------- Auth Gate ----------
_auth_box = st.empty()
with _auth_box.container():
    if not require_access_code():
        st.stop()
_auth_box.empty()

# ---------- Data Load ----------
try:
    data = load_data()
except Exception as e:
    st.error(f"Data load failure: {e}")
    st.stop()

master_df    = data["master_returns"]
longs_df     = data["longs"]
shorts_df    = data["shorts"]
portfolio_df = data["portfolio"]
# factors_df   = data["factors"]
# themes_df    = data["themes"]

# ---------- CSS ----------
st.markdown(
    """
<style>
.block-container {padding-top:2.2rem !important; padding-bottom:2rem;}
[data-testid="stSidebar"] .stSelectbox label {font-weight:600; font-size:0.85rem;}
.page-header-icon {font-size:42px; line-height:1;}
.block-container hr {margin-top:0.6rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Sections ----------
SECTIONS = {
    "Risk Metrics": {
        "Overview": "risk_overview",
        "Risk Metric Tables": "risk_metric_tables",
        "Quadrants": "risk_quadrants",
        "Contributions": "risk_contributions",
        "Heatmap": "risk_heatmap",
    },
    "Clustering Analysis": {
        "Overview": "clustering_overview",
        "Hierarchical Clustering": "cluster_hierarchical",
        "Contributions": "cluster_contributions",
    },
    "Network Analysis": {
        "Overview": "network_overview",
        "Centrality": "network_centrality",
        "Communities": "network_communities",
    },
    #"Factor Analysis": {
        #"Factor Analysis Overview": "fa_overview",
        #"SHAP Analysis": "fa_shap"
    #},
}

# ---------- Navigation State ----------
if "last_page_per_section" not in st.session_state:
    st.session_state.last_page_per_section = {}
if "active_section" not in st.session_state:
    st.session_state.active_section = list(SECTIONS.keys())[0]
if "active_page" not in st.session_state:
    first_sec = st.session_state.active_section
    st.session_state.active_page = list(SECTIONS[first_sec].values())[0]

def _resolve_first_page(section_name: str):
    pages = SECTIONS.get(section_name, {})
    return list(pages.values())[0] if pages else ""

def _on_section_change():
    prev_section = st.session_state.active_section
    st.session_state.last_page_per_section[prev_section] = st.session_state.active_page
    new_section = st.session_state.nav_section
    st.session_state.active_section = new_section
    st.session_state.active_page = st.session_state.last_page_per_section.get(
        new_section, _resolve_first_page(new_section)
    )

def _on_page_change():
    current_section = st.session_state.active_section
    label = st.session_state.nav_page
    if current_section not in SECTIONS:
        st.session_state.active_section = list(SECTIONS.keys())[0]
        current_section = st.session_state.active_section
    st.session_state.active_page = SECTIONS[current_section][label]

SECTIONS.pop("Factor Analysis", None)

# ---------- Sidebar Navigation ----------
st.sidebar.title("📂 Navigation")
section_names = list(SECTIONS.keys())
st.sidebar.selectbox(
    "Section",
    section_names,
    index=section_names.index(st.session_state.active_section),
    key="nav_section",
    on_change=_on_section_change,
)

current_section_pages = SECTIONS[st.session_state.active_section]
page_labels = list(current_section_pages.keys())
page_ids = list(current_section_pages.values())

if st.session_state.active_page not in page_ids:
    st.session_state.active_page = page_ids[0]
current_label = page_labels[page_ids.index(st.session_state.active_page)]

st.sidebar.selectbox(
    "Page",
    page_labels,
    index=page_labels.index(current_label),
    key="nav_page",
    on_change=_on_page_change,
)

st.sidebar.markdown("---")
st.sidebar.caption("Demo • RiskApp")

SECTIONS.pop("Factor Analysis", None)


# ---------- Header Helper ----------
def page_header(icon: str, title: str, subtitle: str = ""):
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.08, 0.92])
    with c1:
        st.markdown(f"<div class='page-header-icon'>{icon}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"### {title}")
        if subtitle:
            st.caption(subtitle)
    st.markdown("---")

# ---------- Page Functions ----------
def risk_overview():
    page_header("","Overview", "")
    st.markdown(
        "<span style='color:red; font-weight:700;'>NOTE: fixed portfolio weights are assumed over historical periods.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("""
The **Risk Metrics** module transforms raw return and portfolio–weight data into a set of complementary views that together give you a full picture of both **descriptive** and **tail** risk, **drawdowns**, **cross‑metric positioning**, and **per‑asset** or **sector** risk contributions.

---

### Risk Metrics Tables  
Computes three side‑by‑side tables for any look‑back window (6 M, 1 Y, etc.) and selected book (FullHist, Long, Short):  
- **Descriptive metrics**:  
  - MAD (Mean Absolute Deviation)  
  - Semi Dev (Semi‑Deviation)  
- **Tail‑risk metrics**:  
  - VaR95, CVaR95, EVaR95, RLVaR95  
  - TG95 (Tail‑Gini), WR (Win Rate), LPM1, LPM2  
- **Drawdown metrics**:  
  - MDD_Abs (Max Drawdown), ADD_Abs (Average Drawdown)  
  - DaR_Abs95 (Drawdown at Risk), CDaR_Abs95 (Conditional DaR)  
  - UCI_Abs (Ulcer Index)  

Each table is annotated with security name, weight, letting you pinpoint both broad statistical behavior and extreme‑tail exposures.

---

### Quadrants  
A customizable scatter‑plot that lets you pick any two metrics (e.g. CVaR95 vs ADD_Abs).  
- **Axes**: any descriptive, tail‑risk or drawdown metric.  
- **Bubble sizing**: by net weight.  
- **Color‑coding**: by sector or long/short category.  
- **Medians**: dashed lines show cross‑sectional medians for easy benchmarking.

---

### Contributions  
Breaks down portfolio risk into **per‑asset** and **sector** contributions.  
- **Windowed**: choose 6 M, 1 Y or 3 Y.  
- **Measures**: CVaR, CDaR, EVaR or MAD.  
- **Signed vs absolute**: toggle raw vs absolute percent contributions.  
- **Sector views**: total and stacked long/short breakdowns.

---

### Rolling Heatmap  
Time‑series heatmap of any metric (VaR95, CVaR95, MDD_Abs, ADD_Abs, etc.) at monthly endpoints.  
- **Trend detection**: color intensity flags regime shifts.  
- **Cross‑asset comparison**: spot co‑movements or outliers.  
- **Interactive**: hover for values; tune alpha, window, frequency, colorscale in the sidebar.

---

This enables you to move from summary tables to cross-metric analysis, dynamic heatmaps, and contribution attribution,
supporting with insights to proactively monitor, diagnose, and adjust portfolio risk.
"""
    )

def clustering_overview():
    page_header("","Clustering – Overview", "How Hierarchical Clustering + Cluster RC help manage risk")
    st.markdown(
        "<span style='color:red; font-weight:700;'>NOTE: fixed portfolio weights are assumed over historical periods.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("""

**Hierarchical Clustering (Spearman + Ward Linkage)**  
- Groups names that *co-move* the most.  
- Interactive heatmap shows correlation strength; magenta boxes ring each cluster.  
- Dendrogram shows the same tree so you can see *when* names merge.  
- A membership table lists tickers, sectors, book (L/S), and intra‑cluster average ρ.

**How to use:**  
- Spot **crowded trades**: big, tight blocks = high common risk.  
- See **hidden factor tilts**: clusters that mix sectors/countries hint at style or macro drivers.  
- Compare **books/windows**: flip Long vs Short or 6M vs 1Y to see if diversification deteriorated.

---

**Cluster Risk Contributions (CVaR / CDaR)**  
- Converts per‑name RC% into **cluster RC%** (weights are gross so longs/shorts don’t cancel).  
- Bar chart ranks clusters by % of total tail risk; table shows top members & cluster weight %.  

**How to use it**  
- **Trim or hedge** the top RC clusters first.  
- Check if **weight concentration ≠ risk concentration** (e.g., small weight, big RC = stealth risk).  
- Track **shifts over time** by changing the window.
"""
    )

def network_overview():
    page_header("","Network Analysis – Overview", "")
    st.markdown(
        "<span style='color:red; font-weight:700;'>NOTE: fixed portfolio weights are assumed over historical periods.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("""
**Why Network Analysis?**  
Traditional correlation tables hide structure. By turning correlations into a graph, you can *see* how names transmit risk, who sits at the center of flows, and which sub-groups move together.

---

### Centrality Tools  
We build a Minimum Spanning Tree (MST) from |Spearman ρ| and compute classic network centrality scores for each node:

- **Degree / Betweenness / Closeness / Eigenvector / PageRank**  
  These quantify how *connected* or *influential* a name is within the co-movement network.

**Value**  
- Spot **risk hubs**: names whose shocks propagate widely (trim/hedge first).  
- Identify **bridge stocks** linking otherwise separate themes (diversify with care).  
- Detect **peripheral positions** that may truly diversify the book.

You can:  
- Color nodes by sector, book, or any centrality metric.  
- Size nodes by **gross weight** to relate importance vs network influence.  
- Hover to see full meta (name, sector, weights, metrics).

---

### Community Detection  
We run modularity-based algorithms (Louvain, Greedy, Label Propagation, Girvan–Newman) on the same MST to find **communities** (tight groups of co-moving names).

**Value:**  
- Discover **hidden factor clusters** that cut across sectors/regions.  
- Measure **cluster concentration** (gross weight % and average |ρ|).  
- Prioritize de-risking: reduce exposure to the largest, most correlated communities.

You can:  
- Color the graph by community, sector, or book.  
- Review a **community summary table** (% gross wt, size, avg |ρ|, top members).  
- Drill into a full membership table for exact constituents.
"""
    )

# ---------- Factor Analysis Pages ----------
#def fa_overview():
    #page_header("", "Factor Analysis Overview", "")

    #st.markdown(
        #"<span style='color:red; font-weight:700;'>"
        #"FOR ILLUSTRATIVE PURPOSES ONLY – Data here are limited by historical coverage, frequency gaps in factor inputs, and access constraints."
        #"</span>",
        #unsafe_allow_html=True,
    #)

    #st.markdown("""
#The **Factor Analysis** module combines a robust XGBoost regression with SHAP‑based explainability to turn factor and secular megatrend theme exposures into insights on portfolio returns.

#---

### Model Training and Validation  
#- **Time‑series split**: holds out the most recent 20% of data to assess out‑of‑sample performance.  
#- **Hyperparameter tuning**: runs a RandomizedSearchCV over tree depth, learning rate, subsample, and colsample parameters using time‑series CV.  

#---

### Global Explanations  
#- **Feature importance**: a bar chart of mean|SHAP| ranks factors by average impact on predictions.  
#- **Beeswarm**: shows the full distribution of each factor’s SHAP values, colored by factor level, to surface non‑linearities and dispersion.

#---

### Local and Temporal Insights  
#- **Dependence plots**: illustrate how the SHAP contribution of one factor varies with its value, optionally colored by a second factor to reveal interactions.  
#- **Time‑series heatmap**: aggregates mean SHAP contributions monthly (or daily) to enable tracking of regime shifts and evolving factor influence.

#---

#With this suite of tools and without data and resource limitations, it would be possible predict returns from factor exposures, but also **understand** exactly BOTH why and **when** those factors drive your portfolio performance.
#""")


#def fa_shap():
    #factor_analysis.render_shap_analysis(master_df, factors_df, themes_df)

# ---------- Router ----------
ROUTER = {
    "risk_overview": risk_overview,
    "risk_metric_tables": lambda: risk_metrics_tables.render(
        master_df, longs_df, shorts_df, portfolio_df, page_header
    ),
    "risk_quadrants": lambda: quadrants.render_quadrants(
        master_df,
        longs_df,
        shorts_df,
        portfolio_df,
        risk_metrics_tables.build_and_split,
        page_header,
    ),
    "risk_contributions": lambda: contributions.render_contributions(
        master_df, longs_df, shorts_df, portfolio_df, page_header
    ),
    "risk_heatmap": lambda: heatmap.render(master_df, page_header),
    "clustering_overview": clustering_overview,
    "cluster_hierarchical": lambda: hierarchical.render_hierarchical(
        master_df, longs_df, shorts_df, portfolio_df, page_header
    ),
    "cluster_contributions": lambda: cluster_contributions.render_cluster_contributions(
        master_df, longs_df, shorts_df, portfolio_df, page_header
    ),
    "network_overview": network_overview,
    "network_centrality": lambda: centrality.render_network(
        master_df, longs_df, shorts_df, portfolio_df, page_header
    ),
    "network_communities": lambda: communities.render_communities(
        master_df, longs_df, shorts_df, portfolio_df, page_header
    ),
    #"fa_overview": fa_overview,
    #"fa_shap": fa_shap
}

# If an old session has a stale section/page, reset it to a valid one
if st.session_state.get("active_section") not in SECTIONS:
    st.session_state.active_section = list(SECTIONS.keys())[0]
if st.session_state.get("active_page") not in SECTIONS[st.session_state.active_section].values():
    st.session_state.active_page = list(SECTIONS[st.session_state.active_section].values())[0]
    
# ---------- Render Selected Page ----------
page_id = st.session_state.active_page
ROUTER.get(page_id, lambda: st.error(f"Unknown page id: {page_id}"))()

# ---------- Footer ----------
st.markdown(
    "<div style='text-align:center; font-size:12px; opacity:.5; margin-top:2rem;'>"
    "Risk Demo App © 2025"
    "</div>",
    unsafe_allow_html=True,
)