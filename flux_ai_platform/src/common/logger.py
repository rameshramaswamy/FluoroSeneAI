import logging
import structlog
import sys
from pythonjsonlogger import jsonlogger

def configure_logger(service_name: str, level: str = "INFO"):
    """
    Configures structured JSON logging for production (ELK/Datadog ready)
    and pretty console logging for development.
    """
    
    # Basic Config
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout)
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()  # Production: JSON output
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    log = structlog.get_logger(service_name)
    return log