import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import inspect
from functools import wraps
import traceback

# ANSI color codes for terminal output
COLORS = {
    "RESET": "\033[0m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "GRAY": "\033[37m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m"
}

# Log levels with colors
LOG_COLORS = {
    "DEBUG": COLORS["GRAY"],
    "INFO": COLORS["BLUE"],
    "WARNING": COLORS["YELLOW"],
    "ERROR": COLORS["RED"],
    "CRITICAL": COLORS["RED"] + COLORS["BOLD"]
}

class APILogFormatter(logging.Formatter):
    """Custom formatter that handles API request/response logging with colors"""

    def format(self, record):
        # Save original message
        original_msg = record.msg

        # Apply color based on level
        levelname = record.levelname
        if hasattr(record, 'color'):
            color = record.color
        else:
            color = LOG_COLORS.get(levelname, COLORS["RESET"])

        # Format the message
        if hasattr(record, 'api_type'):
            api_type = record.api_type
            if api_type == "request":
                prefix = f"{COLORS['BOLD']}{COLORS['CYAN']}API REQUEST{COLORS['RESET']} → "
            elif api_type == "response":
                prefix = f"{COLORS['BOLD']}{COLORS['GREEN']}API RESPONSE{COLORS['RESET']} ← "
            else:
                prefix = ""

            # For API logs, handle JSON formatting
            if isinstance(record.msg, dict):
                try:
                    # Pretty-print the JSON with 2-space indentation
                    formatted_json = json.dumps(record.msg, indent=2)
                    # Limit JSON output length if too large
                    if len(formatted_json) > 500 and record.levelno <= logging.INFO:
                        record.msg = f"{formatted_json[:500]}...\n[truncated, full details in log file]"
                    else:
                        record.msg = formatted_json
                except Exception:
                    # If can't format as JSON, use as is
                    pass

            record.msg = f"{prefix}{record.msg}"

        # Apply color formatting for terminal
        if color and not getattr(record, 'no_color', False):
            record.msg = f"{color}{record.msg}{COLORS['RESET']}"

        # Call the original formatter
        result = super().format(record)

        # Restore original message for future handlers
        record.msg = original_msg
        return result

class DepthTrackingFilter(logging.Filter):
    """Filter that adds depth tracking to log messages"""

    def __init__(self, name=''):
        super().__init__(name)
        self.current_depth = 0
        self.depths = {}  # Track depths by thread

    def filter(self, record):
        # Add indent based on depth
        depth = self.depths.get(record.thread, 0)
        record.depth = depth
        if not hasattr(record, 'no_indent') or not record.no_indent:
            indent = '  ' * depth
            record.msg = f"{indent}{record.msg}"
        return True

    def increase_depth(self, thread_id=None):
        """Increase the indentation depth for a thread"""
        if thread_id is None:
            thread_id = threading.current_thread().ident
        self.depths[thread_id] = self.depths.get(thread_id, 0) + 1

    def decrease_depth(self, thread_id=None):
        """Decrease the indentation depth for a thread"""
        if thread_id is None:
            thread_id = threading.current_thread().ident
        if thread_id in self.depths and self.depths[thread_id] > 0:
            self.depths[thread_id] -= 1

class TimingLogHandler(logging.Handler):
    """Log handler that tracks timing of operations"""

    def __init__(self):
        super().__init__()
        self.start_times = {}
        self.operation_stats = {}

    def emit(self, record):
        # Not actually emitting logs, just tracking timing data
        pass

    def start_operation(self, operation_name):
        """Start timing an operation"""
        self.start_times[operation_name] = time.time()

    def end_operation(self, operation_name):
        """End timing an operation and record stats"""
        if operation_name in self.start_times:
            duration = time.time() - self.start_times[operation_name]
            if operation_name not in self.operation_stats:
                self.operation_stats[operation_name] = {
                    'count': 0,
                    'total_time': 0,
                    'min_time': float('inf'),
                    'max_time': 0
                }

            stats = self.operation_stats[operation_name]
            stats['count'] += 1
            stats['total_time'] += duration
            stats['min_time'] = min(stats['min_time'], duration)
            stats['max_time'] = max(stats['max_time'], duration)

            return duration
        return None

    def get_stats(self):
        """Get current timing statistics"""
        result = {}
        for op_name, stats in self.operation_stats.items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            result[op_name] = {
                'count': stats['count'],
                'total_time': stats['total_time'],
                'avg_time': avg_time,
                'min_time': stats['min_time'] if stats['min_time'] != float('inf') else 0,
                'max_time': stats['max_time']
            }
        return result

# Import threading separately to avoid issues if it's not available
try:
    import threading
except ImportError:
    # Create a minimal thread placeholder if threading is not available
    class threading:
        @staticmethod
        def current_thread():
            class Thread:
                ident = 0
            return Thread()

class EnhancedLogger(logging.Logger):
    """Enhanced logger with API and depth tracking capabilities"""

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.depth_filter = DepthTrackingFilter()
        self.addFilter(self.depth_filter)
        self.timing_handler = TimingLogHandler()
        self.addHandler(self.timing_handler)

    def increase_depth(self):
        """Increase the log indentation depth"""
        self.depth_filter.increase_depth()

    def decrease_depth(self):
        """Decrease the log indentation depth"""
        self.depth_filter.decrease_depth()

    def api_request(self, message, level=logging.DEBUG):
        """Log an API request"""
        if self.isEnabledFor(level):
            record = self.makeRecord(
                self.name, level, "", 0, message, (), None,
                extra={'api_type': 'request'}
            )
            self.handle(record)

    def api_response(self, message, level=logging.DEBUG):
        """Log an API response"""
        if self.isEnabledFor(level):
            record = self.makeRecord(
                self.name, level, "", 0, message, (), None,
                extra={'api_type': 'response'}
            )
            self.handle(record)

    def start_operation(self, operation_name):
        """Start timing an operation"""
        self.timing_handler.start_operation(operation_name)
        self.debug(f"▶ Starting operation: {operation_name}")
        self.increase_depth()

    def end_operation(self, operation_name):
        """End timing an operation"""
        duration = self.timing_handler.end_operation(operation_name)
        self.decrease_depth()
        if duration is not None:
            self.debug(f"✓ Completed operation: {operation_name} in {duration:.2f}s")

    def get_timing_stats(self):
        """Get timing statistics"""
        return self.timing_handler.get_stats()

def setup_enhanced_logger(level: str = 'INFO',
                          file: Optional[str] = None,
                          api_log_file: Optional[str] = None,
                          console: bool = True) -> EnhancedLogger:
    """
    Set up an enhanced logger with API request/response tracking and depth visualization.

    Args:
        level: Logging level ('DEBUG', 'INFO', etc.)
        file: Main log file path
        api_log_file: Separate log file for API requests/responses
        console: Whether to log to console

    Returns:
        Enhanced logger configured with the specified handlers
    """
    # Register our custom logger class
    logging.setLoggerClass(EnhancedLogger)

    # Create logger instance
    logger = logging.getLogger('jobbot')
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a detailed formatter for general logs
    main_fmt = APILogFormatter(
        '%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )

    # Create a simplified formatter for API logs
    api_fmt = APILogFormatter(
        '%(asctime)s [API] %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(main_fmt)
        logger.addHandler(console_handler)

    # Main file handler
    if file:
        # Make sure parent directory exists
        Path(file).parent.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(main_fmt)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {file}")

    # API log file handler (if specified)
    if api_log_file:
        # Make sure parent directory exists
        Path(api_log_file).parent.mkdir(exist_ok=True, parents=True)
        api_handler = logging.FileHandler(api_log_file)
        api_handler.setFormatter(api_fmt)

        # Only log API-related messages to this handler
        class APIFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'api_type')

        api_handler.addFilter(APIFilter())
        logger.addHandler(api_handler)
        logger.info(f"API logging to file: {api_log_file}")

    # Test log message
    logger.debug("Enhanced logger initialized")

    return logger

class DepthContext:
    """Context manager for tracking depth in logs"""

    def __init__(self, logger, label=None, log_level='DEBUG'):
        self.logger = logger
        self.label = label
        self.log_level = getattr(logging, log_level, logging.DEBUG)
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.label:
            self.logger.log(self.log_level, f"▶ STARTED: {self.label}")
        self.logger.increase_depth()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.decrease_depth()
        if self.label:
            duration = time.perf_counter() - self.start_time
            self.logger.log(self.log_level, f"✓ COMPLETED: {self.label} in {duration:.2f}s")

        if exc_type:
            self.logger.error(f"ERROR in {self.label}: {exc_val}")
            self.logger.debug(f"Traceback: {''.join(traceback.format_exception(exc_type, exc_val, exc_tb))}")

def trace_api_calls(logger):
    """
    Decorator factory that creates decorators for tracing API calls.
    Wraps functions to log the request/response of API interactions.

    Args:
        logger: The logger to use for API call tracing

    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get function info
            func_name = func.__qualname__

            # Get caller info for better context
            caller_frame = inspect.currentframe().f_back
            caller_info = ""
            if caller_frame:
                caller_info = f" from {caller_frame.f_code.co_name}()"

            # Log the API request
            request_data = {
                "function": func_name,
                "args": str(args) if args else None,
                "kwargs": kwargs if kwargs else None
            }
            logger.api_request(request_data)

            try:
                # Call the original function
                with DepthContext(logger, f"API Call: {func_name}{caller_info}"):
                    result = await func(*args, **kwargs)

                # Log the API response
                response_summary = result
                # If result is too large, create a summary
                if isinstance(result, dict) and len(str(result)) > 1000:
                    response_summary = {
                        "summary": "Large response truncated",
                        "type": str(type(result)),
                        "keys": list(result.keys()) if hasattr(result, "keys") else None
                    }

                logger.api_response(response_summary)
                return result
            except Exception as e:
                # Log API errors
                error_data = {
                    "error": str(e),
                    "type": type(e).__name__,
                    "function": func_name
                }
                logger.api_response(error_data, level=logging.ERROR)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get function info
            func_name = func.__qualname__

            # Get caller info for better context
            caller_frame = inspect.currentframe().f_back
            caller_info = ""
            if caller_frame:
                caller_info = f" from {caller_frame.f_code.co_name}()"

            # Log the API request
            request_data = {
                "function": func_name,
                "args": str(args) if args else None,
                "kwargs": kwargs if kwargs else None
            }
            logger.api_request(request_data)

            try:
                # Call the original function
                with DepthContext(logger, f"API Call: {func_name}{caller_info}"):
                    result = func(*args, **kwargs)

                # Log the API response
                response_summary = result
                # If result is too large, create a summary
                if isinstance(result, dict) and len(str(result)) > 1000:
                    response_summary = {
                        "summary": "Large response truncated",
                        "type": str(type(result)),
                        "keys": list(result.keys()) if hasattr(result, "keys") else None
                    }

                logger.api_response(response_summary)
                return result
            except Exception as e:
                # Log API errors
                error_data = {
                    "error": str(e),
                    "type": type(e).__name__,
                    "function": func_name
                }
                logger.api_response(error_data, level=logging.ERROR)
                raise

        # Return the appropriate wrapper based on whether the function is async or not
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
