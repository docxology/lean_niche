#!/usr/bin/env python3
"""
LeanNiche Comprehensive Logging Configuration

This module provides centralized logging configuration for all Lean operations,
with detailed tracking of Lean steps, proof verification, and mathematical analysis.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


class LeanNicheFormatter(logging.Formatter):
    """Custom formatter for LeanNiche logging with enhanced Lean step tracking"""

    def __init__(self, include_lean_context: bool = True):
        super().__init__()
        self.include_lean_context = include_lean_context

    def format(self, record: logging.LogRecord) -> str:
        # Add Lean-specific context
        if hasattr(record, 'lean_module'):
            record.msg = f"[LEAN:{record.lean_module}] {record.msg}"
        if hasattr(record, 'theorem_name'):
            record.msg = f"[THEOREM:{record.theorem_name}] {record.msg}"
        if hasattr(record, 'proof_step'):
            record.msg = f"[STEP:{record.proof_step}] {record.msg}"

        # Standard formatting
        formatted = super().format(record)

        # Add Lean execution context if available
        if self.include_lean_context and hasattr(record, 'lean_context'):
            context_info = f"\nLEAN Context: {record.lean_context}"
            formatted += context_info

        return formatted


class LeanStepFilter(logging.Filter):
    """Filter for Lean-specific log records"""

    def __init__(self, lean_only: bool = False):
        super().__init__()
        self.lean_only = lean_only

    def filter(self, record: logging.LogRecord) -> bool:
        if self.lean_only:
            return hasattr(record, 'lean_module') or 'lean' in record.name.lower()
        return True


def get_logging_config(log_level: str = "INFO",
                      log_to_file: bool = True,
                      log_to_console: bool = True,
                      enable_lean_tracking: bool = True) -> Dict[str, Any]:
    """Get comprehensive logging configuration for LeanNiche"""

    # Ensure logs directory exists
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)

    # Convert log level string to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    numeric_level = level_map.get(log_level.upper(), logging.INFO)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "lean_detailed": {
                "()": LeanNicheFormatter,
                "include_lean_context": enable_lean_tracking,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "lean_compact": {
                "format": "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
            },
            "lean_json": {
                "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
            }
        },
        "filters": {
            "lean_steps": {
                "()": LeanStepFilter,
                "lean_only": False
            },
            "lean_only": {
                "()": LeanStepFilter,
                "lean_only": True
            }
        },
        "handlers": {},
        "loggers": {
            "lean_niche": {
                "level": numeric_level,
                "handlers": [],
                "propagate": False,
                "filters": ["lean_steps"]
            },
            "lean_niche.core": {
                "level": numeric_level,
                "handlers": [],
                "propagate": True,
                "filters": ["lean_steps"]
            },
            "lean_niche.lean": {
                "level": numeric_level,
                "handlers": [],
                "propagate": True,
                "filters": ["lean_only"]
            },
            "lean_niche.analysis": {
                "level": numeric_level,
                "handlers": [],
                "propagate": True,
                "filters": ["lean_steps"]
            },
            "lean_niche.proof_verification": {
                "level": numeric_level,
                "handlers": [],
                "propagate": True,
                "filters": ["lean_only"]
            },
            "lean_niche.testing": {
                "level": numeric_level,
                "handlers": [],
                "propagate": True,
                "filters": ["lean_steps"]
            }
        },
        "root": {
            "level": numeric_level,
            "handlers": [],
            "filters": ["lean_steps"]
        }
    }

    # Configure handlers based on options
    handlers = []

    if log_to_console:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "lean_detailed",
            "stream": "ext://sys.stdout",
            "filters": ["lean_steps"]
        }
        handlers.append("console")

    if log_to_file:
        # Main log file
        config["handlers"]["file_main"] = {
            "class": "logging.FileHandler",
            "formatter": "lean_detailed",
            "filename": str(logs_dir / "lean_niche.log"),
            "encoding": "utf-8",
            "filters": ["lean_steps"]
        }

        # Lean-specific log file
        config["handlers"]["file_lean"] = {
            "class": "logging.FileHandler",
            "formatter": "lean_detailed",
            "filename": str(logs_dir / "lean_operations.log"),
            "encoding": "utf-8",
            "filters": ["lean_only"]
        }

        # JSON log file for structured analysis
        config["handlers"]["file_json"] = {
            "class": "logging.FileHandler",
            "formatter": "lean_json",
            "filename": str(logs_dir / "lean_niche.json"),
            "encoding": "utf-8"
        }

        handlers.extend(["file_main", "file_lean", "file_json"])

    # Add handlers to loggers
    for logger_name in config["loggers"]:
        config["loggers"][logger_name]["handlers"] = handlers

    config["root"]["handlers"] = handlers

    return config


def setup_lean_logging(log_level: str = "INFO",
                      enable_file_logging: bool = True,
                      enable_console_logging: bool = True,
                      enable_lean_tracking: bool = True) -> logging.Logger:
    """Setup comprehensive logging for LeanNiche"""

    config = get_logging_config(
        log_level=log_level,
        log_to_file=enable_file_logging,
        log_to_console=enable_console_logging,
        enable_lean_tracking=enable_lean_tracking
    )

    logging.config.dictConfig(config)

    # Get the main logger
    logger = logging.getLogger("lean_niche")

    # Log setup completion
    logger.info("LeanNiche logging system initialized", extra={
        "lean_context": {
            "log_level": log_level,
            "file_logging": enable_file_logging,
            "console_logging": enable_console_logging,
            "lean_tracking": enable_lean_tracking,
            "timestamp": datetime.now().isoformat()
        }
    })

    return logger


def create_lean_logger(module_name: str,
                      lean_module: Optional[str] = None) -> logging.Logger:
    """Create a logger specifically for Lean operations"""

    logger = logging.getLogger(f"lean_niche.{module_name}")

    # Add Lean-specific context if provided
    if lean_module:
        logger = logging.LoggerAdapter(logger, {"lean_module": lean_module})

    return logger


def log_lean_step(logger: logging.Logger,
                 step_name: str,
                 lean_module: str,
                 theorem_name: Optional[str] = None,
                 status: str = "started",
                 details: Optional[Dict[str, Any]] = None):
    """Log a specific Lean step with comprehensive context"""

    extra = {
        "lean_module": lean_module,
        "proof_step": step_name,
        "step_status": status,
        "timestamp": datetime.now().isoformat()
    }

    if theorem_name:
        extra["theorem_name"] = theorem_name

    if details:
        extra["lean_context"] = details

    logger.info(f"Lean step {status}: {step_name}", extra=extra)


def log_lean_result(logger: logging.Logger,
                   operation: str,
                   lean_module: str,
                   success: bool,
                   result_details: Optional[Dict[str, Any]] = None,
                   error_details: Optional[Dict[str, Any]] = None):
    """Log Lean operation results with detailed context"""

    level = logging.INFO if success else logging.ERROR
    status = "succeeded" if success else "failed"

    extra = {
        "lean_module": lean_module,
        "operation_status": status,
        "timestamp": datetime.now().isoformat()
    }

    if result_details:
        extra["result_details"] = result_details

    if error_details:
        extra["error_details"] = error_details

    message = f"Lean operation {status}: {operation}"

    if success:
        logger.log(level, message, extra=extra)
    else:
        logger.log(level, message, extra=extra)


def log_lean_verification(logger: logging.Logger,
                         theorem_name: str,
                         lean_module: str,
                         verification_result: Dict[str, Any]):
    """Log Lean theorem verification results"""

    success = verification_result.get("success", False)
    level = logging.INFO if success else logging.WARNING

    extra = {
        "lean_module": lean_module,
        "theorem_name": theorem_name,
        "verification_success": success,
        "timestamp": datetime.now().isoformat(),
        "lean_context": verification_result
    }

    status = "VERIFIED" if success else "FAILED"
    logger.log(level, f"Theorem {status}: {theorem_name}", extra=extra)


class LeanLogger:
    """Enhanced logger specifically for Lean operations"""

    def __init__(self, module_name: str, lean_module: Optional[str] = None):
        self.logger = create_lean_logger(module_name, lean_module)
        self.lean_module = lean_module or module_name
        self.active_steps = {}

    def log_step_start(self, step_name: str, details: Optional[Dict[str, Any]] = None):
        """Log the start of a Lean step"""
        self.active_steps[step_name] = datetime.now()
        log_lean_step(self.logger, step_name, self.lean_module,
                     status="started", details=details)

    def log_step_end(self, step_name: str, success: bool = True,
                    result_details: Optional[Dict[str, Any]] = None):
        """Log the completion of a Lean step"""
        start_time = self.active_steps.pop(step_name, None)
        duration = None

        if start_time:
            duration = (datetime.now() - start_time).total_seconds()
            if result_details is None:
                result_details = {}
            result_details["duration_seconds"] = duration

        log_lean_step(self.logger, step_name, self.lean_module,
                     status="completed" if success else "failed",
                     details=result_details)

    def log_verification(self, theorem_name: str, result: Dict[str, Any]):
        """Log theorem verification result"""
        log_lean_verification(self.logger, theorem_name, self.lean_module, result)

    def log_error(self, operation: str, error: Exception,
                 additional_context: Optional[Dict[str, Any]] = None):
        """Log Lean operation error with full context"""
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "operation": operation
        }

        if additional_context:
            error_details.update(additional_context)

        self.logger.error(f"Lean error in {operation}: {error}", extra={
            "lean_module": self.lean_module,
            "error_details": error_details,
            "timestamp": datetime.now().isoformat()
        })

    def log_performance(self, operation: str, duration: float,
                       metrics: Optional[Dict[str, Any]] = None):
        """Log performance metrics for Lean operations"""
        perf_details = {
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        }

        if metrics:
            perf_details.update(metrics)

        self.logger.info(f"Lean performance: {operation} completed in {duration:.3f}s",
                        extra={
                            "lean_module": self.lean_module,
                            "performance_metrics": perf_details
                        })


# Global logger instance for convenience
_main_logger = None

def get_main_logger() -> logging.Logger:
    """Get the main LeanNiche logger"""
    global _main_logger
    if _main_logger is None:
        _main_logger = setup_lean_logging()
    return _main_logger


def log_system_status(logger: logging.Logger,
                     component: str,
                     status: str,
                     details: Optional[Dict[str, Any]] = None):
    """Log system component status"""
    extra = {
        "component": component,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }

    if details:
        extra["component_details"] = details

    logger.info(f"System status: {component} is {status}", extra=extra)


# Initialize logging when module is imported
def _initialize_module_logging():
    """Initialize logging when this module is first imported"""
    try:
        # Only setup if not already configured
        if not logging.getLogger().handlers:
            setup_lean_logging(log_level="INFO")
    except Exception as e:
        # Fallback to basic logging if configuration fails
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).warning(f"Could not initialize enhanced logging: {e}")


# Initialize on import
_initialize_module_logging()
