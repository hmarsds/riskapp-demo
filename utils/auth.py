# utils/auth.py
import streamlit as st

def require_access_code():
    expected = st.secrets.get("APP_PASS")
    if not expected:
        return True  # no password configured
    if st.session_state.get("authed"):
        return True

    # Render inside a container so we can clear it later
    code = st.text_input("Enter access code", type="password")
    if code:
        if code == expected:
            st.session_state.authed = True
            st.success("Access granted.")
            return True
        else:
            st.error("Incorrect code.")
    return False