#!/usr/bin/env python3
"""
Comprehensive tests for LeanNiche logging configuration and Lean step tracking.
"""

import pytest
import logging
import tempfile
import json
from pathlib import Path
from datetime import datetime

from src.python.core.logging_config import (
    get_logging_config,
    setup_lean_logging,
    LeanLogger,
    create_lean_logger,
    log_lean_step,
    log_lean_result,
    log_lean_verification,
    LeanNicheFormatter,
    LeanStepFilter
)


class TestLeanNicheFormatter:
    """Test the custom LeanNiche formatter"""

    def test_formatter_initialization(self):
        """Test formatter initialization with and without Lean context"""
        formatter = LeanNicheFormatter(include_lean_context=True)
        assert formatter.include_lean_context is True

        formatter = LeanNicheFormatter(include_lean_context=False)
        assert formatter.include_lean_context is False

    def test_formatter_with_lean_context(self):
        """Test formatter with Lean context information"""
        formatter = LeanNicheFormatter(include_lean_context=True)

        # Create a mock record with Lean context
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.lean_module = "TestModule"
        record.lean_context = {"test": "data"}

        formatted = formatter.format(record)
        assert "[LEAN:TestModule]" in formatted
        assert '"test": "data"' in formatted


class TestLeanStepFilter:
    """Test the Lean step filter"""

    def test_lean_only_filter(self):
        """Test filter that only allows Lean-related records"""
        lean_filter = LeanStepFilter(lean_only=True)

        # Record without Lean attributes
        regular_record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Regular message", args=(), exc_info=None
        )
        assert not lean_filter.filter(regular_record)

        # Record with Lean module
        lean_record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Lean message", args=(), exc_info=None
        )
        lean_record.lean_module = "TestModule"
        assert lean_filter.filter(lean_record)

    def test_all_filter(self):
        """Test filter that allows all records"""
        all_filter = LeanStepFilter(lean_only=False)

        regular_record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Regular message", args=(), exc_info=None
        )
        assert all_filter.filter(regular_record)


class TestLoggingConfig:
    """Test logging configuration functionality"""

    def test_get_logging_config_basic(self):
        """Test basic logging configuration generation"""
        config = get_logging_config(log_level="INFO")

        assert "version" in config
        assert "handlers" in config
        assert "loggers" in config
        assert "formatters" in config

        # Check Lean-specific formatters
        assert "lean_detailed" in config["formatters"]
        assert "lean_compact" in config["formatters"]
        assert "lean_json" in config["formatters"]

        # Check Lean-specific filters
        assert "lean_steps" in config["filters"]
        assert "lean_only" in config["filters"]

    def test_get_logging_config_with_files(self, tmp_path):
        """Test logging configuration with file handlers"""
        config = get_logging_config(
            log_level="DEBUG",
            log_to_file=True,
            log_to_console=False
        )

        handlers = config["handlers"]
        assert "file_main" in handlers
        assert "file_lean" in handlers
        assert "file_json" in handlers

        # Check file paths contain logs directory
        assert "logs" in handlers["file_main"]["filename"]

    def test_setup_lean_logging(self, tmp_path):
        """Test setting up Lean logging"""
        logger = setup_lean_logging(log_level="INFO")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "lean_niche"

        # Test that loggers for different modules exist
        core_logger = logging.getLogger("lean_niche.core")
        lean_logger = logging.getLogger("lean_niche.lean")
        analysis_logger = logging.getLogger("lean_niche.analysis")

        assert core_logger.parent == logger
        assert lean_logger.parent == logger
        assert analysis_logger.parent == logger


class TestLeanLogger:
    """Test the LeanLogger class"""

    def test_lean_logger_creation(self):
        """Test creating a Lean logger"""
        logger = LeanLogger("test_module", "TestModule")

        assert logger.logger.name == "lean_niche.test_module"
        assert logger.lean_module == "TestModule"

    def test_log_step_tracking(self, caplog):
        """Test logging step start and end"""
        logger = LeanLogger("test_module", "TestModule")

        # Test step start
        logger.log_step_start("test_step", {"param": "value"})

        # Check that step was logged
        assert any("test_step" in record.message for record in caplog.records)
        assert any("started" in record.message for record in caplog.records)

        # Test step end
        logger.log_step_end("test_step", success=True, result_details={"result": "success"})

        # Check that step completion was logged
        assert any("completed" in record.message for record in caplog.records)

    def test_log_verification(self, caplog):
        """Test logging verification results"""
        logger = LeanLogger("test_module", "TestModule")

        verification_result = {
            "success": True,
            "theorems_proven": ["theorem1", "theorem2"],
            "execution_time": 1.5
        }

        logger.log_verification("TestTheorem", verification_result)

        # Check that verification was logged
        assert any("TestTheorem" in record.message for record in caplog.records)
        assert any("VERIFIED" in record.message for record in caplog.records)

    def test_log_error(self, caplog):
        """Test logging errors"""
        logger = LeanLogger("test_module", "TestModule")

        test_error = ValueError("Test error")
        additional_context = {"module": "test", "operation": "test_op"}

        logger.log_error("test_operation", test_error, additional_context)

        # Check that error was logged
        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_records) > 0
        assert any("Test error" in r.message for r in error_records)

    def test_log_performance(self, caplog):
        """Test logging performance metrics"""
        logger = LeanLogger("test_module", "TestModule")

        logger.log_performance("test_operation", 2.5, {"iterations": 100})

        # Check that performance was logged
        assert any("test_operation" in record.message for record in caplog.records)
        assert any("2.500s" in record.message for record in caplog.records)


class TestLeanStepFunctions:
    """Test standalone Lean step logging functions"""

    def test_log_lean_step(self, caplog):
        """Test the log_lean_step function"""
        logger = logging.getLogger("test")
        log_lean_step(logger, "test_step", "TestModule",
                     status="started", details={"key": "value"})

        assert len(caplog.records) > 0
        assert any("test_step" in record.message for record in caplog.records)
        assert any("started" in record.message for record in caplog.records)

    def test_log_lean_result(self, caplog):
        """Test the log_lean_result function"""
        logger = logging.getLogger("test")

        log_lean_result(logger, "test_operation", "TestModule", True,
                       result_details={"theorems": 5})

        assert len(caplog.records) > 0
        assert any("test_operation" in record.message for record in caplog.records)
        assert any("succeeded" in record.message for record in caplog.records)

    def test_log_lean_verification(self, caplog):
        """Test the log_lean_verification function"""
        logger = logging.getLogger("test")

        result = {"success": True, "theorems_proven": ["thm1"]}
        log_lean_verification(logger, "TestTheorem", "TestModule", result)

        assert len(caplog.records) > 0
        assert any("TestTheorem" in record.message for record in caplog.records)


class TestIntegration:
    """Test integration of logging with actual Lean operations"""

    def test_full_lean_workflow_logging(self, tmp_path, caplog):
        """Test complete workflow with logging"""
        # Setup logging
        logger = LeanLogger("integration_test", "IntegrationTest")

        # Simulate a complete Lean workflow
        logger.log_step_start("lean_workflow", {
            "workflow_type": "theorem_proving",
            "expected_theorems": 3
        })

        # Step 1: Code generation
        logger.log_step_start("code_generation")
        # Simulate code generation
        lean_code = "import LeanNiche.Basic\ntheorem test : true := by trivial"
        logger.log_step_end("code_generation", success=True, result_details={
            "code_length": len(lean_code),
            "imports": ["LeanNiche.Basic"]
        })

        # Step 2: Verification
        logger.log_step_start("verification")
        verification_result = {
            "success": True,
            "theorems_proven": ["test"],
            "execution_time": 0.5
        }
        logger.log_verification("test", verification_result)
        logger.log_step_end("verification", success=True)

        # Step 3: Result saving
        logger.log_step_start("save_results")
        saved_files = ["theorems.json", "proofs.lean"]
        logger.log_step_end("save_results", success=True, result_details={
            "files_saved": len(saved_files),
            "file_types": saved_files
        })

        # Complete workflow
        logger.log_step_end("lean_workflow", success=True, result_details={
            "total_steps": 3,
            "theorems_verified": 1,
            "files_generated": len(saved_files)
        })

        # Verify logging captured all steps
        messages = [record.message for record in caplog.records]

        # Check that all major steps were logged
        assert any("lean_workflow" in msg and "started" in msg for msg in messages)
        assert any("code_generation" in msg and "completed" in msg for msg in messages)
        assert any("verification" in msg and "completed" in msg for msg in messages)
        assert any("save_results" in msg and "completed" in msg for msg in messages)
        assert any("lean_workflow" in msg and "completed" in msg for msg in messages)


class TestLeanVerificationValidation:
    """Test that logging properly validates Lean verification"""

    def test_verification_success_logging(self, caplog):
        """Test logging of successful verification"""
        logger = LeanLogger("verification_test", "VerificationTest")

        # Successful verification
        result = {
            "success": True,
            "theorems_proven": ["central_limit_theorem", "law_of_large_numbers"],
            "compilation_successful": True,
            "total_proofs": 2,
            "success_rate": 100.0
        }

        logger.log_verification("CentralLimitTheorem", result)

        # Check logging
        assert len(caplog.records) > 0
        messages = [r.message for r in caplog.records]
        assert any("CentralLimitTheorem" in msg for msg in messages)
        assert any("VERIFIED" in msg for msg in messages)

    def test_verification_failure_logging(self, caplog):
        """Test logging of failed verification"""
        logger = LeanLogger("verification_test", "VerificationTest")

        # Failed verification
        result = {
            "success": False,
            "error": "Compilation failed",
            "stderr": "error: unknown identifier",
            "compilation_successful": False,
            "total_proofs": 0,
            "success_rate": 0.0
        }

        logger.log_verification("InvalidTheorem", result)

        # Check logging
        assert len(caplog.records) > 0
        messages = [r.message for r in caplog.records]
        assert any("InvalidTheorem" in msg for msg in messages)
        assert any("FAILED" in msg for msg in messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
