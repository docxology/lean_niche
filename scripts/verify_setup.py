#!/usr/bin/env python3
"""
LeanNiche Setup Verification Script
Verifies that all components are properly configured and ready for use.

Thin orchestrator: path bootstrap + CLI only. All verification logic lives
in src/python/utils/setup_verifier.py.
"""

import argparse
import sys
from pathlib import Path

# Path bootstrap: make the src package importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.python.utils.setup_verifier import SetupVerifier


def main():
    """Main verification function"""
    parser = argparse.ArgumentParser(
        description="Verify that all LeanNiche components are properly configured."
    )
    parser.parse_args()

    # Navigate to project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent.absolute()

    verifier = SetupVerifier(project_root)
    all_passed, issues, warnings = verifier.run_verification()

    verifier.print_report()

    if issues:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
