import os
import requests
from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, jwk
from jose.utils import base64url_decode
from src.common.logger import configure_logger
from src.common.config import settings

log = configure_logger("auth_middleware")
security = HTTPBearer()

class OIDCGuard:
    def __init__(self):
        self.jwks_url = f"{settings.KEYCLOAK_URL}/realms/{settings.KEYCLOAK_REALM}/protocol/openid-connect/certs"
        self.jwks_cache = None
        self.last_fetch = 0

    def _get_jwks(self):
        """Fetch and cache public keys from Keycloak"""
        now = time.time()
        # Refresh every hour
        if self.jwks_cache is None or (now - self.last_fetch > 3600):
            try:
                response = requests.get(self.jwks_url, timeout=5)
                response.raise_for_status()
                self.jwks_cache = response.json()
                self.last_fetch = now
                log.info("jwks_refreshed_from_keycloak")
            except Exception as e:
                log.error("jwks_fetch_failed", error=str(e))
                # Fallback to hardcoded dev secret if keycloak is unreachable (ONLY IN DEV)
                if settings.ENV == "dev": return None
                raise e
        return self.jwks_cache

    def verify_token(self, credentials: HTTPAuthorizationCredentials = Security(security)):
        token = credentials.credentials
        
        try:
            # 1. Get Header to find Key ID (kid)
            unverified_header = jwt.get_unverified_header(token)
            
            # 2. Get Public Keys
            jwks = self._get_jwks()
            
            if jwks:
                # Find matching key
                rsa_key = {}
                for key in jwks["keys"]:
                    if key["kid"] == unverified_header["kid"]:
                        rsa_key = {
                            "kty": key["kty"],
                            "kid": key["kid"],
                            "use": key["use"],
                            "n": key["n"],
                            "e": key["e"]
                        }
                if not rsa_key:
                    raise HTTPException(status_code=401, detail="Token key ID not found in JWKS")
                
                # 3. Verify Signature (RS256)
                payload = jwt.decode(
                    token,
                    rsa_key,
                    algorithms=["RS256"],
                    audience="flux-platform",
                    issuer=f"{settings.KEYCLOAK_URL}/realms/{settings.KEYCLOAK_REALM}"
                )
            else:
                # Dev Fallback
                payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])

            # 4. Context Extraction
            user_id = payload.get("sub")
            roles = payload.get("realm_access", {}).get("roles", [])
            
            return {"user_id": user_id, "roles": roles, "raw": payload}

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except Exception as e:
            log.warning("auth_failed", error=str(e))
            raise HTTPException(status_code=401, detail="Invalid Authentication")

auth = OIDCGuard()