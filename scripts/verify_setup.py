#!/usr/bin/env python3
"""
LeanNiche Setup Verification Script
Verifies that all components are properly configured and ready for use.
"""

import os
import sys
import toml
import json
from pathlib import Path
from typing import Dict, List, Tuple

class SetupVerifier:
    """Comprehensive setup verification"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues = []
        self.warnings = []

    def verify_file_exists(self, file_path: str, description: str) -> bool:
        """Check if file exists"""
        full_path = self.project_root / file_path
        if not full_path.exists():
            self.issues.append(f"Missing {description}: {file_path}")
            return False
        return True

    def verify_directory_exists(self, dir_path: str, description: str) -> bool:
        """Check if directory exists"""
        full_path = self.project_root / dir_path
        if not full_path.is_dir():
            self.issues.append(f"Missing {description}: {dir_path}")
            return False
        return True

    def verify_executable(self, file_path: str, description: str) -> bool:
        """Check if file is executable"""
        full_path = self.project_root / file_path
        if not os.access(full_path, os.X_OK):
            self.issues.append(f"Not executable {description}: {file_path}")
            return False
        return True

    def verify_lean_files(self) -> bool:
        """Verify Lean source files"""
        lean_dir = self.project_root / "src" / "lean"
        if not lean_dir.exists():
            self.issues.append("Missing src/lean directory")
            return False

        required_files = [
            "Main.lean",
            "Basic.lean",
            "Advanced.lean",
            "Computational.lean",
            "DynamicalSystems.lean",
            "Lyapunov.lean",
            "SetTheory.lean",
            "Statistics.lean",
            "Tactics.lean",
            "Utils.lean",
            "Visualization.lean",
            "Setup.lean"
        ]

        all_present = True
        for lean_file in required_files:
            if not (lean_dir / lean_file).exists():
                self.issues.append(f"Missing Lean file: {lean_file}")
                all_present = False

        return all_present

    def verify_python_files(self) -> bool:
        """Verify Python source files"""
        python_dir = self.project_root / "src" / "python"
        if not python_dir.exists():
            self.issues.append("Missing src/python directory")
            return False

        required_files = [
            "__init__.py",
            "cli.py",
            "visualization.py"
        ]

        all_present = True
        for py_file in required_files:
            if not (python_dir / py_file).exists():
                self.issues.append(f"Missing Python file: {py_file}")
                all_present = False

        return all_present

    def verify_latex_files(self) -> bool:
        """Verify LaTeX conversion files"""
        latex_dir = self.project_root / "src" / "latex"
        if not latex_dir.exists():
            self.issues.append("Missing src/latex directory")
            return False

        required_files = [
            "__init__.py",
            "lean_to_latex.py"
        ]

        all_present = True
        for latex_file in required_files:
            if not (latex_dir / latex_file).exists():
                self.issues.append(f"Missing LaTeX file: {latex_file}")
                all_present = False

        return all_present

    def verify_scripts(self) -> bool:
        """Verify build scripts"""
        scripts_dir = self.project_root / "scripts"
        if not scripts_dir.exists():
            self.issues.append("Missing scripts directory")
            return False

        required_scripts = [
            "build.sh",
            "test.sh",
            "analyze.sh"
        ]

        all_present = True
        for script in required_scripts:
            script_path = scripts_dir / script
            if not script_path.exists():
                self.issues.append(f"Missing script: {script}")
                all_present = False
            elif not os.access(script_path, os.X_OK):
                self.warnings.append(f"Script not executable: {script}")

        return all_present

    def verify_config_files(self) -> bool:
        """Verify configuration files"""
        config_files = [
            "lakefile.toml",
            "pyproject.toml",
            "lean-toolchain",
            ".env.example"
        ]

        all_present = True
        for config_file in config_files:
            if not self.verify_file_exists(config_file, f"configuration file {config_file}"):
                all_present = False

        # Check TOML syntax
        try:
            with open(self.project_root / "lakefile.toml", 'r') as f:
                toml.load(f)
        except Exception as e:
            self.issues.append(f"Invalid lakefile.toml: {e}")

        return all_present

    def verify_structure(self) -> bool:
        """Verify overall project structure"""
        required_dirs = [
            "src",
            "src/lean",
            "src/python",
            "src/latex",
            "src/tests",
            "scripts",
            "docs",
            "examples"
        ]

        all_present = True
        for dir_path in required_dirs:
            if not self.verify_directory_exists(dir_path, f"directory {dir_path}"):
                all_present = False

        return all_present

    def run_verification(self) -> Tuple[bool, List[str], List[str]]:
        """Run complete verification"""
        print("🔍 LeanNiche Setup Verification")
        print("=" * 50)

        # Run all verification checks
        checks = [
            ("Project Structure", self.verify_structure),
            ("Configuration Files", self.verify_config_files),
            ("Lean Source Files", self.verify_lean_files),
            ("Python Source Files", self.verify_python_files),
            ("LaTeX Files", self.verify_latex_files),
            ("Build Scripts", self.verify_scripts),
        ]

        all_passed = True
        for check_name, check_func in checks:
            print(f"\n📋 Checking {check_name}...")
            if check_func():
                print(f"✅ {check_name} - PASSED")
            else:
                print(f"❌ {check_name} - FAILED")
                all_passed = False

        # Additional checks
        print("\n📋 Checking setup script...")
        if self.verify_executable("setup.sh", "setup script"):
            print("✅ Setup script - EXECUTABLE")
        else:
            print("❌ Setup script - NOT EXECUTABLE")
            all_passed = False

        print("\n📋 Checking Makefile...")
        if self.verify_file_exists("Makefile", "Makefile"):
            print("✅ Makefile - PRESENT")
        else:
            print("❌ Makefile - MISSING")
            all_passed = False

        return all_passed, self.issues, self.warnings

    def print_report(self):
        """Print detailed verification report"""
        print("\n📊 VERIFICATION REPORT")
        print("=" * 50)

        if not self.issues and not self.warnings:
            print("🎉 All checks passed! LeanNiche is ready for setup.")
            print("\n🚀 Next steps:")
            print("   1. Run: ./setup.sh")
            print("   2. Or run: make setup")
            print("   3. Follow the on-screen instructions")
        else:
            if self.issues:
                print(f"\n❌ ISSUES FOUND ({len(self.issues)}):")
                for issue in self.issues:
                    print(f"   • {issue}")

            if self.warnings:
                print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
                for warning in self.warnings:
                    print(f"   • {warning}")

            print("\n🔧 Please fix the issues above before running setup.")

def main():
    """Main verification function"""
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
