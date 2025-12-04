import streamlit as st
import requests
import os
import sys

# Add root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.fortress.secrets import vault

# Configuration
API_URL = os.getenv("API_URL", "http://gateway:8080")

st.set_page_config(page_title="Flux AI | Enterprise", layout="wide", page_icon="🏥")

def authenticate_api(username, password):
    """
    Exchange credentials for JWT via Keycloak (Mocked here for prototype simplicity)
    In prod: request.post(KEYCLOAK_TOKEN_ENDPOINT, data={...})
    """
    # Verify against Vault or Keycloak
    if username == "doctor" and password == "flux":
        return "valid.jwt.token"
    return None

def main():
    if "token" not in st.session_state:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.title("🔐 Flux Secure Login")
            with st.form("login_form"):
                user = st.text_input("Username")
                pw = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In")
                
                if submitted:
                    token = authenticate_api(user, pw)
                    if token:
                        st.session_state["token"] = token
                        st.session_state["user"] = user
                        st.success("Authenticated.")
                        st.rerun()
                    else:
                        st.error("Access Denied.")
        return

    # --- Authenticated View ---
    st.sidebar.title(f"👨‍⚕️ Dr. {st.session_state['user']}")
    if st.sidebar.button("Logout"):
        del st.session_state["token"]
        st.rerun()

    st.title("🏥 Patient Worklist")
    
    # 1. Fetch Patient List (Protected API Call)
    # headers = {"Authorization": f"Bearer {st.session_state['token']}"}
    # resp = requests.get(f"{API_URL}/v1/patients", headers=headers)
    
    st.info("System Status: 🟢 Online | Vault: 🔒 Secured | Audit: ✍️ Active")
    
    uploaded_file = st.file_uploader("Upload Fluoroscopy Frame", type=['png', 'jpg'])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Input Frame", width=300)
        if st.button("Analyze (Secure Inference)"):
            with st.spinner("Processing..."):
                # Simulate API call with Audit context
                import time; time.sleep(0.5)
                st.success("Segmentation Complete")
                st.json({
                    "inference_id": "inf_998811",
                    "latency": "24ms",
                    "model_version": "v2.1 (Swin-UNet)"
                })

if __name__ == "__main__":
    main()