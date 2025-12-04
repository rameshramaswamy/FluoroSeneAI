import os
import hvac
import time
from cachetools import TTLCache, cached
from src.common.logger import configure_logger
from src.common.exceptions import FluxError

log = configure_logger("secrets_manager")

class SecretsManager:
    def __init__(self):
        self.vault_url = os.getenv("VAULT_ADDR", "http://localhost:8200")
        # Cache secrets for 5 minutes
        self.cache = TTLCache(maxsize=100, ttl=300)
        self.client = None
        self._authenticate()

    def _authenticate(self):
        """
        Enterprise Auth Strategy:
        1. Try Kubernetes Auth (Production)
        2. Try Token Auth (Dev/CI)
        """
        try:
            self.client = hvac.Client(url=self.vault_url)
            
            # Scenario A: Kubernetes (Production)
            if os.getenv("KUBERNETES_PORT"):
                f = open('/var/run/secrets/kubernetes.io/serviceaccount/token')
                jwt = f.read()
                self.client.auth_kubernetes("flux-role", jwt)
                log.info("vault_auth_success", method="kubernetes")
            
            # Scenario B: Token (Dev)
            else:
                token = os.getenv("VAULT_TOKEN", "root")
                self.client.token = token
                if self.client.is_authenticated():
                    log.info("vault_auth_success", method="token")
                else:
                    log.warning("vault_auth_failed_invalid_token")

        except Exception as e:
            log.error("vault_connection_error", error=str(e))
            self.client = None

    def get_secret(self, path: str, key: str, default: str = None) -> str:
        """
        Fetches secret with Caching and Fallbacks.
        """
        cache_key = f"{path}:{key}"
        
        # 1. Check Cache
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 2. Try Vault
        if self.client:
            try:
                response = self.client.secrets.kv.v2.read_secret_version(path=path)
                secret_val = response['data']['data'].get(key)
                if secret_val:
                    self.cache[cache_key] = secret_val # Cache it
                    return secret_val
            except Exception as e:
                # If Vault is down, log warning but try env vars
                log.warning("vault_lookup_failed", path=path, error=str(e))

        # 3. Fallback to Env Var
        env_val = os.getenv(key.upper())
        if env_val:
            return env_val

        # 4. Default
        if default:
            return default

        raise FluxError(f"Critical Security Failure: Secret {key} not found anywhere.")

vault = SecretsManager()