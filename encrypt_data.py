#!/usr/bin/env python3
"""
Encrypt four raw data files from an external source directory (argonaut_data)
into ./data_private/*.enc using a Fernet key.

Usage:
  python encrypt_data.py
        -> generates a NEW key (prints it)

  DATA_ENCRYPTION_KEY=<existing_key> python encrypt_data.py
        -> reuses that key (no new key printed)

After running, put the key into .streamlit/secrets.toml as:
DATA_ENCRYPTION_KEY = "<key>"

If 'portfolio_data.xlsx' is actually parquet, adjust the code where noted.
"""
from cryptography.fernet import Fernet, InvalidToken
from pathlib import Path
import os, io, hashlib, sys
import pandas as pd

# ---- Adjust these if needed ----
SOURCE_DIR = Path.home() / "Desktop" / "argonaut_data"
OUT_DIR = Path(__file__).resolve().parent / "data_private"
OUT_DIR.mkdir(exist_ok=True)

FILES = [
    ("master_returns.parquet",  "parquet"),
    ("longs_df_gbp.parquet",    "parquet"),
    ("short_df_gbp.parquet",    "parquet"),
    ("portfolio_data.xlsx",     "excel"),   # change to ("portfolio_data.parquet","parquet") if not Excel
]

def get_key():
    k = os.getenv("DATA_ENCRYPTION_KEY")
    if k:
        return k, False
    k = Fernet.generate_key().decode()
    return k, True

def serialize(path: Path, kind: str) -> bytes:
    if kind == "parquet":
        df = pd.read_parquet(path)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue()
    elif kind == "excel":
        df = pd.read_excel(path)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False)
        return buf.getvalue()
    else:
        raise ValueError(f"Unsupported kind: {kind}")

def short_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def main():
    key, generated = get_key()
    fernet = Fernet(key.encode())
    print("\n================ ENCRYPTION KEY ================\n"
          f"{key}\n"
          "================================================")
    print(("Generated NEW key." if generated else "Re-using existing key.")
          + " Save as DATA_ENCRYPTION_KEY in .streamlit/secrets.toml\n")

    successes, skips = 0, []
    for fname, kind in FILES:
        src = SOURCE_DIR / fname
        if not src.exists():
            print(f"[SKIP] {fname} (not found at {src})")
            skips.append(fname)
            continue
        raw = serialize(src, kind)
        token = fernet.encrypt(raw)
        out_name = f"{Path(fname).stem}.enc"
        out_path = OUT_DIR / out_name
        out_path.write_bytes(token)

        # verify
        try:
            if fernet.decrypt(token) != raw:
                print(f"[FAIL] Verification mismatch for {fname}")
                continue
        except InvalidToken:
            print(f"[FAIL] InvalidToken while verifying {fname}")
            continue

        print(f"[OK]  {fname} -> {out_name} "
              f"plainSHA={short_hash(raw)} encSHA={short_hash(token)} "
              f"rawKB={len(raw)/1024:.1f}")
        successes += 1

    print(f"\nSummary: {successes} encrypted, {len(skips)} skipped.")
    if skips:
        print("Skipped:", ", ".join(skips))
    if successes == 0:
        sys.exit(1)

if __name__ == "__main__":
    main()