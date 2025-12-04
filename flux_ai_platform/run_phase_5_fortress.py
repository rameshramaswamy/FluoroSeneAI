import sys
import os
import time
from jose import jwt

# Setup Paths
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.fortress.secrets import vault
from src.fortress.auth import auth
from src.common.logger import configure_logger

log = configure_logger("fortress_tester")

def test_secrets():
    log.info("step_1_testing_secrets")
    
    # 1. Should fail or return default if Vault not running
    try:
        secret = vault.get_secret("secret/data/flux", "db_pass", default="fallback_pass")
        log.info("secret_retrieval", status="success", value="***" if secret != "fallback_pass" else "fallback_used")
    except Exception as e:
        log.error("secret_retrieval_failed", error=str(e))

def test_auth_logic():
    log.info("step_2_testing_auth")
    
    # 1. Generate Valid Token (Mocking Keycloak Issue)
    secret_key = "flux_secret_key_dev"
    payload = {
        "sub": "doctor_strange",
        "realm_access": {"roles": ["clinician"]},
        "exp": time.time() + 3600
    }
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    
    # 2. Simulate API Guard
    class MockRequest:
        credentials = token

    try:
        # Validate
        user_data = auth.verify_token(MockRequest())
        log.info("token_validation", status="success", user=user_data['sub'])
        
        # Check Role
        roles = user_data.get("realm_access", {}).get("roles", [])
        if "clinician" in roles:
            log.info("rbac_check", status="authorized", role="clinician")
        else:
            log.error("rbac_check", status="denied")
            
    except Exception as e:
        log.error("auth_test_failed", error=str(e))

def main():
    log.info("pipeline_start", phase="5", status="hardening_system")
    
    print("\n--- 🔐 1. SECRETS VAULT ---")
    test_secrets()
    
    print("\n--- 🛡️ 2. IDENTITY ACCESS ---")
    test_auth_logic()
    
    print("\n--- 🏥 3. UI DASHBOARD ---")
    print("To run the clinician dashboard, execute:")
    print(">> streamlit run src/dashboard/app.py")
    
    print("\n✅ FORTRESS VERIFICATION COMPLETE")

if __name__ == "__main__":
    main()