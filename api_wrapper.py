"""
API wrapper module for OpenAI API calls with logging.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from openai import OpenAI

# Global client instance
_api_wrapper = None

class APIWrapper:
    """Wrapper around OpenAI API with logging and timing"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.client = OpenAI()

        # Store API call statistics
        self.stats = {
            "total_calls": 0,
            "success_calls": 0,
            "error_calls": 0,
            "total_time": 0.0,
        }

    def _log_api_call(self, method: str, success: bool, duration: float, **kwargs):
        """Log an API call with details"""
        status = "SUCCESS" if success else "ERROR"
        self.logger.debug(f"API {method} - {status} in {duration:.2f}s")

        # Update stats
        self.stats["total_calls"] += 1
        self.stats["total_time"] += duration

        if success:
            self.stats["success_calls"] += 1
        else:
            self.stats["error_calls"] += 1

        # Log detailed call info to API log if available
        if hasattr(self.logger, 'api_log'):
            # Format the call parameters as a log entry
            params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.api_log.info(f"{method} - {status} - {duration:.2f}s - {params}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current API usage statistics"""
        return self.stats

def initialize_api_wrapper(logger: logging.Logger) -> None:
    """Initialize the API wrapper with the provided logger"""
    global _api_wrapper
    _api_wrapper = APIWrapper(logger)

def get_api_wrapper() -> APIWrapper:
    """Get the initialized API wrapper instance"""
    if _api_wrapper is None:
        raise RuntimeError("API wrapper not initialized. Call initialize_api_wrapper first.")
    return _api_wrapper
