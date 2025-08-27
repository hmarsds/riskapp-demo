from pathlib import Path
import io, os
import pandas as pd
from cryptography.fernet import Fernet, InvalidToken
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "data_private"

def _get_key() -> bytes:
    # secrets first, then env
    key = None
    try:
        key = st.secrets.get("DATA_ENCRYPTION_KEY")
    except Exception:
        pass
    if not key:
        key = os.getenv("DATA_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("DATA_ENCRYPTION_KEY not set (secrets or env).")
    return key.strip().encode()

def _dec(name: str) -> bytes:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing encrypted file: {path}")
    token = path.read_bytes()
    try:
        return Fernet(_get_key()).decrypt(token)
    except InvalidToken:
        raise RuntimeError(f"Decryption failed (InvalidToken) for {name}. "
                           "Key mismatch or file corrupted.")

@st.cache_data(show_spinner="Decrypting & loading datasets...")
def load_data():
    # Decrypt into memory
    master_bytes    = _dec("master_returns.enc")
    longs_bytes     = _dec("longs_df_gbp.enc")
    shorts_bytes    = _dec("short_df_gbp.enc")
    portfolio_bytes = _dec("portfolio_data.enc")

    master_df  = pd.read_parquet(io.BytesIO(master_bytes))
    longs_df   = pd.read_parquet(io.BytesIO(longs_bytes))
    shorts_df  = pd.read_parquet(io.BytesIO(shorts_bytes))

    # If portfolio file is Excel; otherwise change to pd.read_parquet(...)
    try:
        portfolio_df = pd.read_excel(io.BytesIO(portfolio_bytes))
    except Exception:
        # fallback parquet (uncomment if needed)
        # portfolio_df = pd.read_parquet(io.BytesIO(portfolio_bytes))
        raise

    # --- Unencrypted parquet for factor/theme analysis ---
    #factors_path = DATA_DIR / "factors.parquet"
    #themes_path  = DATA_DIR / "themes.parquet"

    #if not factors_path.exists():
        #raise FileNotFoundError(f"Missing factors file: {factors_path}")
    #if not themes_path.exists():
        #raise FileNotFoundError(f"Missing themes file: {themes_path}")

    #factors_df = pd.read_parquet(factors_path)
    #themes_df  = pd.read_parquet(themes_path)


    return {
        "master_returns": master_df,
        "longs": longs_df,
        "shorts": shorts_df,
        "portfolio": portfolio_df,
        #"factors": factors_df,
        #"themes": themes_df,
    }