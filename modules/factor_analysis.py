import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import plotly.express as px

@st.cache_data(show_spinner="Training XGBoost + SHAP…")
def fit_and_explain(df: pd.DataFrame,
                    date_col: str,
                    target_col: str,
                    feature_cols: list[str]):
    # 1️⃣ Prepare data
    df = df.dropna(subset=[target_col] + feature_cols).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    X = df[feature_cols]
    y = df[target_col]

    # 2️⃣ Hyper‐parameter tuning via RandomizedSearch
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }
    base = xgb.XGBRegressor(random_state=42, objective="reg:squarederror")
    search = RandomizedSearchCV(base, param_dist,
                                n_iter=20, cv=3,
                                scoring="neg_root_mean_squared_error",
                                random_state=42,
                                n_jobs=-1)
    search.fit(X, y)
    model = search.best_estimator_

    # 3️⃣ SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    # compute RMSE manually for compatibility
    mse = mean_squared_error(y, model.predict(X))
    rmse = mse ** 0.5

    return model, X, y, shap_vals, explainer, rmse




def train_xgb_shap_tuned(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    param_distributions: dict = None,
    n_iter: int = 20,
    cv_splits: int = 5,
    **fixed_xgb_kwargs
):
    # ——— Step 0: Prepare data ———
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    # ——— Step 1: Train/test split ———
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )

    # ——— Step 2: Hyperparameter tuning ———
    if param_distributions is None:
        param_distributions = {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
    base = xgb.XGBRegressor(random_state=random_state, **fixed_xgb_kwargs)
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    # ——— Step 3: SHAP on hold-out ———
    explainer = shap.TreeExplainer(best)
    shap_vals = explainer.shap_values(X_test)
    shap_df = pd.DataFrame(shap_vals, index=X_test.index, columns=feature_cols)

    return best, X_test, y_test, shap_df, explainer


def render_shap_analysis(master_df, factors_df, themes_df):
    """
    Renders SHAP analysis using hold-out from train_xgb_shap_tuned.
    """
    # Apply white text styling for Matplotlib
    plt.rcParams.update({
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.edgecolor': 'white',
        'axes.facecolor': 'none'
    })

    # Sidebar controls
    """
    Renders SHAP analysis using hold-out from train_xgb_shap_tuned.
    """
    # Sidebar controls
    st.sidebar.markdown("### SHAP Analysis Dataset")
    choice = st.sidebar.radio("Run SHAP on:", ["Factors", "Themes"])
    df = factors_df if choice == "Factors" else themes_df

    # Derive feature columns
    feature_cols = [c for c in df.columns if c not in ("date", "Portfolio Returns")]

    st.header(f"📊 SHAP Analysis on {choice}")

    # 1️⃣ Train, tune, and get hold-out results
    best_model, X_test, y_test, shap_df, explainer = train_xgb_shap_tuned(
        df=df,
        date_col="date",
        target_col="Portfolio Returns",
        feature_cols=feature_cols,
        test_size=0.2,
        random_state=42,
        param_distributions={
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        },
        n_iter=30,
        cv_splits=4,
    )

    # 2️⃣ SHAP Feature Importance
    st.subheader("SHAP Feature Importance")
    plt.figure()
    shap.summary_plot(
        shap_df.values,
        X_test,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # 3️⃣ SHAP Beeswarm Plot
    st.subheader("SHAP Beeswarm Plot")
    plt.figure()
    shap.summary_plot(
        shap_df.values,
        X_test,
        plot_type="dot",
        show=False
    )
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # 4️⃣ SHAP Dependence Plot
    st.subheader("SHAP Dependence Plot")
    feat = st.sidebar.selectbox("Feature to plot", feature_cols, index=0)
    interaction = st.sidebar.selectbox("Color by (interaction)", ["None"] + feature_cols, index=0)
    plt.figure()
    shap.dependence_plot(
        feat,
        explainer.shap_values(X_test),
        X_test,
        interaction_index=None if interaction == "None" else interaction,
        show=False
    )
    title_suffix = f" (colored by {interaction})" if interaction != "None" else ""
    plt.title(f"Dependence: {feat}{title_suffix}")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # 5️⃣ Time-series SHAP Heatmap
    st.subheader("Time-series SHAP Heatmap")
    st.sidebar.markdown("#### SHAP heatmap settings")
    freq = st.sidebar.radio("Aggregation", ["M", "D"], index=0)
    shap_hold = pd.DataFrame(
        explainer.shap_values(X_test),
        columns=feature_cols,
        index=X_test.index
    )
    agg = shap_hold.resample(freq).mean()
    fig = px.imshow(
        agg.T,
        labels=dict(x="Date", y="Feature", color="Mean SHAP"),
        aspect="auto",
        origin="lower",
        template="plotly_dark",
        color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(fig, use_container_width=True)
