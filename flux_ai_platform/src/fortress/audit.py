import time
import uuid
from fastapi import Request
from src.common.logger import configure_logger

# Dedicated logger for audit trails (routed to separate file/index in ELK)
audit_log = configure_logger("hipaa_audit_trail")

class AuditMiddleware:
    """
    Middleware to log every request for compliance.
    Records: Who, When, What (Path), Result (Status).
    """
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Capture User Info if Auth Middleware ran first
        # (FastAPI middleware runs safely even if Auth fails)
        user_identity = "anonymous"
        
        try:
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000
            
            # Extract user from request state if available (set by Auth)
            if hasattr(request.state, "user"):
                user_identity = request.state.user.get("user_id", "unknown")

            audit_log.info(
                "access_event",
                trace_id=request_id,
                user=user_identity,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=process_time,
                client_ip=request.client.host
            )
            
            return response
            
        except Exception as e:
            audit_log.error(
                "access_error",
                trace_id=request_id,
                user=user_identity,
                path=request.url.path,
                error=str(e)
            )
            raise e