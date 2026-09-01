"""
Supabase auth helpers for Portfolio Comparator.
"""

import streamlit as st
import os

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")


def get_client() -> "Client | None":
    if not SUPABASE_AVAILABLE or not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def init_auth_state():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None


def render_auth_sidebar():
    """Renders login/signup/logout UI in the sidebar. Returns current user or None."""
    init_auth_state()
    sb = get_client()

    if sb is None:
        st.sidebar.caption("🔒 Auth unavailable — set SUPABASE_URL and SUPABASE_ANON_KEY env vars.")
        return None

    if st.session_state.user:
        email = st.session_state.user.get("email", "")
        st.sidebar.success(f"Signed in as **{email}**")
        if st.sidebar.button("Sign Out", use_container_width=True):
            try:
                sb.auth.sign_out()
            except Exception:
                pass
            st.session_state.user = None
            st.session_state.auth_token = None
            st.rerun()
        return st.session_state.user

    # Login / Sign Up tabs
    login_tab, signup_tab = st.sidebar.tabs(["Sign In", "Sign Up"])

    with login_tab:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        if st.button("Sign In", use_container_width=True, key="btn_signin"):
            if email and password:
                try:
                    resp = sb.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user = {"id": resp.user.id, "email": resp.user.email}
                    st.session_state.auth_token = resp.session.access_token
                    st.rerun()
                except Exception as e:
                    st.error(f"Sign in failed: {e}")
            else:
                st.warning("Enter email and password.")

    with signup_tab:
        new_email = st.text_input("Email", key="signup_email")
        new_pw = st.text_input("Password (min 6 chars)", type="password", key="signup_pw")
        if st.button("Create Account", use_container_width=True, key="btn_signup"):
            if new_email and len(new_pw) >= 6:
                try:
                    resp = sb.auth.sign_up({"email": new_email, "password": new_pw})
                    if resp.user:
                        st.success("Account created! Check your email to confirm, then sign in.")
                    else:
                        st.error("Sign up failed — try a different email.")
                except Exception as e:
                    st.error(f"Sign up failed: {e}")
            else:
                st.warning("Enter a valid email and password (6+ chars).")

    return None


# ── Saved portfolios ──────────────────────────────────────────────────────────

def save_portfolio(user_id: str, name: str, weights: dict, auth_token: str):
    sb = get_client()
    if sb is None:
        return False
    try:
        sb.postgrest.auth(auth_token)
        sb.table("saved_portfolios").upsert({
            "user_id": user_id,
            "name": name,
            "weights": weights,
        }, on_conflict="user_id,name").execute()
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False


def load_portfolios(user_id: str, auth_token: str) -> list[dict]:
    sb = get_client()
    if sb is None:
        return []
    try:
        sb.postgrest.auth(auth_token)
        resp = sb.table("saved_portfolios").select("name,weights,created_at").eq("user_id", user_id).order("created_at", desc=True).execute()
        return resp.data or []
    except Exception:
        return []


def delete_portfolio(user_id: str, name: str, auth_token: str):
    sb = get_client()
    if sb is None:
        return
    try:
        sb.postgrest.auth(auth_token)
        sb.table("saved_portfolios").delete().eq("user_id", user_id).eq("name", name).execute()
    except Exception as e:
        st.error(f"Delete failed: {e}")
