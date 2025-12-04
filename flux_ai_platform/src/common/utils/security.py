import hashlib
import uuid

def generate_trace_id() -> str:
    """Generates a unique ID for tracking requests across microservices"""
    return str(uuid.uuid4())

def calculate_sha256(file_bytes: bytes) -> str:
    """Calculates SHA256 hash for data integrity verification"""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(file_bytes)
    return sha256_hash.hexdigest()