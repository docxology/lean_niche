#!/usr/bin/env python3
"""
Lean Code Runner and Result Extractor

This module provides utilities to run Lean code, extract results,
and integrate Lean proofs with Python analysis tools.
"""

import subprocess
import tempfile
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime
from .logging_config import LeanLogger, log_lean_step, log_lean_result, log_lean_verification


class LeanRunner:
    """Runner for Lean code execution and result extraction"""

    def __init__(self, lean_path: str = "lean", timeout: int = 30, lean_module: Optional[str] = None):
        """Initialize the Lean runner with enhanced logging"""
        self.lean_path = lean_path
        self.timeout = timeout
        self.lean_module = lean_module or "LeanRunner"
        self.lean_logger = LeanLogger("core.lean_runner", self.lean_module)
        self.logger = self.lean_logger.logger

        # Setup logging
        self.setup_logging()

        # Log initialization
        self.lean_logger.logger.info(f"LeanRunner initialized with path: {lean_path}, timeout: {timeout}s")

    def setup_logging(self):
        """Setup logging configuration"""
        import os
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # Ensure logs directory exists relative to project root
        project_root = Path(__file__).resolve().parents[3]
        logs_dir = project_root / 'logs'
        logs_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(logs_dir / 'lean_runner.log'),
                logging.StreamHandler()
            ]
        )

    def run_lean_code(self, code: str, imports: List[str] = None) -> Dict[str, Any]:
        """Run Lean code with comprehensive logging and extract results"""
        if imports is None:
            imports = []

        # Log the operation start with detailed context
        operation_details = {
            "code_length": len(code),
            "imports_count": len(imports),
            "imports": imports,
            "lean_path": self.lean_path,
            "timeout": self.timeout
        }

        self.lean_logger.log_step_start("run_lean_code", operation_details)

        temp_file = None
        result = None
        try:
            # Step 1: Create temporary file
            self.lean_logger.log_step_start("create_temp_file")

            with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
                # Write imports with logging
                for imp in imports:
                    f.write(f"import {imp}\n")
                    self.lean_logger.logger.debug(f"Added import: {imp}")

                # Write the code
                f.write(code)
                from .lean_source import iter_lean_namespace_scopes
                declarations = []
                for line, namespaces, _ in iter_lean_namespace_scopes(code):
                    match = re.match(r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:public|private|protected)\s+)?(theorem|lemma)\s+([A-Za-z][A-Za-z0-9_']*)\b", line)
                    if match:
                        kind, name = match.groups()
                        qualified = ".".join((*namespaces, name))
                        declarations.append((kind, qualified))
                        f.write(f"\n#print axioms _root_.{qualified}\n")

                temp_file = Path(f.name)

            self.lean_logger.log_step_end("create_temp_file", success=True, result_details={
                "temp_file": str(temp_file),
                "file_size": len(code)
            })

            # Step 2: Execute Lean
            self.lean_logger.log_step_start("execute_lean", {
                "temp_file": str(temp_file),
                "lean_command": self.lean_path
            })

            result = self._execute_lean(temp_file)

            self.lean_logger.log_step_end("execute_lean", success=True, result_details={
                "execution_time": result.get('execution_time', 0),
                "return_code": result.get('returncode', -1),
                "stdout_length": len(result.get('stdout', '')),
                "stderr_length": len(result.get('stderr', ''))
            })

            # Step 3: Parse results
            self.lean_logger.log_step_start("parse_results", {
                "stdout_length": len(result.get('stdout', '')),
                "stderr_length": len(result.get('stderr', ''))
            })

            parsed_result = self._parse_lean_output(result, code)
            # Only kernel dependency reports can certify a declaration.
            traces = {}
            for match in re.finditer(r"'([^']+)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)", result.get('stdout', ''), re.S):
                traces[match.group(1)] = set(filter(None, re.split(r"[,\s]+", match.group(2) or '')))
            standard = {'propext', 'Classical.choice', 'Quot.sound'}
            verified = []
            rejected = []
            for kind, name in declarations:
                dependencies = traces.get(name)
                if dependencies is None or not dependencies <= standard:
                    rejected.append(name)
                else:
                    verified.append({'type': kind, 'name': name, 'axioms': sorted(dependencies), 'method': 'kernel_axiom_report'})
            status = parsed_result['verification_status']
            if rejected:
                status['compilation_successful'] = False
                parsed_result['error_details'].append({'message': 'Missing or nonstandard kernel evidence', 'declarations': rejected})
            successful = status['compilation_successful']
            parsed_result['theorems_proven'] = [d for d in verified if d['type'] == 'theorem'] if successful else []
            parsed_result['lemmas_proven'] = [d for d in verified if d['type'] == 'lemma'] if successful else []
            status['total_proofs'] = len(verified) if successful else 0
            status['verification_complete'] = successful
            status['success_rate'] = (100 * len(verified) / len(declarations)) if successful and declarations else 0
            status['kernel_policy_successful'] = successful and not rejected
            status['kernel_axiom_reports'] = traces


            self.lean_logger.log_step_end("parse_results", success=True, result_details={
                "theorems_found": len(parsed_result.get('theorems_proven', [])),
                "definitions_found": len(parsed_result.get('definitions_created', [])),
                "lemmas_found": len(parsed_result.get('lemmas_proven', [])),
                "verification_status": parsed_result.get('verification_status', {})
            })

            # Step 4: Log final verification results
            verification_status = parsed_result.get('verification_status', {})
            log_lean_result(
                self.lean_logger.logger,
                "run_lean_code",
                self.lean_module,
                verification_status.get('compilation_successful', False),
                result_details={
                    "theorems_proven": len(parsed_result.get('theorems_proven', [])),
                    "compilation_success": verification_status.get('compilation_successful', False),
                    "total_proofs": verification_status.get('total_proofs', 0),
                    "success_rate": verification_status.get('success_rate', 0)
                }
            )

            self.lean_logger.log_step_end("run_lean_code", success=True, result_details={
                "total_theorems": len(parsed_result.get('theorems_proven', [])),
                "compilation_successful": verification_status.get('compilation_successful', False),
                "verification_complete": verification_status.get('verification_complete', False)
            })

            return {
                'success': verification_status.get('compilation_successful', False),
                **({"error": result.get("stderr") or result.get("stdout") or "Lean proof verification was rejected"}
                   if not verification_status.get('compilation_successful', False) else {}),
                'result': parsed_result,
                'stdout': result.get('stdout', ''),
                'stderr': result.get('stderr', ''),
                'execution_time': result.get('execution_time', 0)
            }

        except Exception as e:
            # Log error with full context
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "temp_file": str(temp_file) if temp_file else None,
                "imports": imports,
                "stdout": result.get('stdout', '') if result else '',
                "stderr": result.get('stderr', '') if result else ''
            }

            self.lean_logger.log_error("run_lean_code", e, error_details)
            self.lean_logger.log_step_end("run_lean_code", success=False, result_details=error_details)

            return {
                'success': False,
                'error': str(e),
                'stdout': result.get('stdout', '') if result else '',
                'stderr': result.get('stderr', '') if result else '',
                'execution_time': result.get('execution_time', 0) if result else 0
            }

        finally:
            # Clean up temporary file
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                    self.lean_logger.logger.debug(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    self.lean_logger.logger.warning(f"Failed to clean up temp file {temp_file}: {e}")

    def _execute_lean(self, lean_file: Path) -> Dict[str, Any]:
        """Execute Lean file and capture output"""
        start_time = datetime.now()
        try:
            # Run Lean from the project root so that module search paths and
            # lake build outputs (.olean) are discoverable. Also extend LEAN_PATH
            # to include local `src/lean` and any `.lake/**/build` directories.
            project_root = Path(__file__).resolve().parents[3]

            # Build LEAN_PATH environment
            env = os.environ.copy()
            lean_path_entries = []
            # Resolve compiled interfaces before source directories. Lean stops
            # at a matching source module even when its adjacent .olean is absent.
            project_build = project_root / '.lake' / 'build' / 'lib' / 'lean'
            if project_build.is_dir():
                lean_path_entries.append(str(project_build))
            for p in sorted(project_root.glob('.lake/**/build/lib/lean')):
                if p.is_dir() and p != project_build:
                    lean_path_entries.append(str(p))
            src_lean = project_root / 'src' / 'lean'
            if src_lean.is_dir():
                lean_path_entries.append(str(src_lean))

            # prepend to existing LEAN_PATH
            existing_lp = env.get('LEAN_PATH', '')
            combined = os.pathsep.join([*lean_path_entries, existing_lp]) if existing_lp else os.pathsep.join(lean_path_entries)
            env['LEAN_PATH'] = combined

            if not lean_file.is_file():
                raise FileNotFoundError(lean_file)

            result = subprocess.run(
                [self.lean_path, str(lean_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(project_root),
                env=env
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time
            }

        except subprocess.TimeoutExpired:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': f'Timeout after {self.timeout} seconds',
                'execution_time': execution_time
            }

    def _parse_lean_output(self, result: Dict[str, Any], source_code: str = None) -> Dict[str, Any]:
        """Parse Lean output and extract comprehensive proof outcomes"""
        parsed = {
            'theorems_proven': [],
            'definitions_created': [],
            'examples_verified': [],
            'lemmas_proven': [],
            'axioms_defined': [],
            'inductive_types': [],
            'structures_defined': [],
            'namespaces_created': [],
            'tactics_used': [],
            'proof_methods': [],
            'mathematical_properties': [],
            'computation_results': [],
            'verification_status': [],
            'performance_metrics': [],
            'error_details': [],
            'warning_details': [],
            'compilation_info': [],
            'dependency_info': []
        }

        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')

        # Parse stdout for comprehensive results
        lines = stdout.split('\n')
        current_context = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Extract compilation information
            if 'Lean' in line and 'version' in line.lower():
                parsed['compilation_info'].append(line)
            elif 'compiling' in line.lower() or 'building' in line.lower():
                parsed['compilation_info'].append(line)

            # Extract theorem proofs with comprehensive details
            theorem_match = re.search(r'^(theorem|lemma|axiom)\s+(\w+)\s*[:.]', line)
            if theorem_match:
                proof_type = theorem_match.group(1)
                proof_name = theorem_match.group(2)
                proof_info = {
                    'type': proof_type,
                    'name': proof_name,
                    'line': line,
                    'context': current_context
                }
                if proof_type == 'theorem':
                    parsed['theorems_proven'].append(proof_info)
                elif proof_type == 'lemma':
                    parsed['lemmas_proven'].append(proof_info)
                elif proof_type == 'axiom':
                    parsed['axioms_defined'].append(proof_info)

            # Extract definitions with types
            def_match = re.search(r'^(def|definition|constant|abbrev)\s+(\w+)\s*[:.]', line)
            if def_match:
                def_type = def_match.group(1)
                def_name = def_match.group(2)
                def_info = {
                    'type': def_type,
                    'name': def_name,
                    'line': line,
                    'context': current_context
                }
                parsed['definitions_created'].append(def_info)

            # Extract inductive types
            inductive_match = re.search(r'^(inductive|structure|class)\s+(\w+)', line)
            if inductive_match:
                type_kind = inductive_match.group(1)
                type_name = inductive_match.group(2)
                type_info = {
                    'kind': type_kind,
                    'name': type_name,
                    'line': line,
                    'context': current_context
                }
                if type_kind == 'inductive':
                    parsed['inductive_types'].append(type_info)
                elif type_kind == 'structure':
                    parsed['structures_defined'].append(type_info)

            # Extract namespace declarations
            namespace_match = re.search(r'^(namespace|section|end)\s+(\w+)', line)
            if namespace_match:
                ns_type = namespace_match.group(1)
                ns_name = namespace_match.group(2)
                if ns_type == 'namespace':
                    parsed['namespaces_created'].append({
                        'name': ns_name,
                        'line': line
                    })
                    current_context = ns_name
                elif ns_type in ['end', 'section']:
                    current_context = None

            # Extract examples and their verification
            example_match = re.search(r'^(example|#eval|#check|#reduce)', line)
            if example_match:
                example_info = {
                    'type': example_match.group(1),
                    'line': line,
                    'context': current_context,
                    'verified': 'no_errors' in line.lower() or 'success' in line.lower()
                }
                parsed['examples_verified'].append(example_info)

            # Extract proof tactics used
            tactic_match = re.search(r'\b(by|with|using)\s+([a-zA-Z_][a-zA-Z0-9_]*\b)', line)
            if tactic_match:
                tactic_info = {
                    'keyword': tactic_match.group(1),
                    'tactic': tactic_match.group(2),
                    'line': line,
                    'context': current_context
                }
                parsed['tactics_used'].append(tactic_info)

            # Extract mathematical properties proven
            property_patterns = [
                (r'commutative', 'commutativity'),
                (r'associative', 'associativity'),
                (r'distributive', 'distributivity'),
                (r'idempotent', 'idempotence'),
                (r'inverse', 'inverse_property'),
                (r'identity', 'identity_property'),
                (r'stable', 'stability'),
                (r'continuous', 'continuity'),
                (r'differentiable', 'differentiability'),
                (r'integrable', 'integrability'),
                (r'convergent', 'convergence'),
                (r'bounded', 'boundedness')
            ]

            for pattern, prop_type in property_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    parsed['mathematical_properties'].append({
                        'type': prop_type,
                        'pattern': pattern,
                        'line': line,
                        'context': current_context
                    })

            # Extract computation results
            if any(keyword in line.lower() for keyword in ['=', 'result:', 'value:', 'output:']):
                parsed['computation_results'].append({
                    'line': line,
                    'context': current_context
                })

        # Parse stderr for detailed error and warning analysis
        error_lines = (stdout + '\n' + stderr).split('\n')
        for line in error_lines:
            line = line.strip()
            if not line:
                continue

            if re.search(r'\berror(?:\([^)]*\))?:', line, re.IGNORECASE):
                error_info = self._parse_error_line(line)
                parsed['error_details'].append(error_info)
            elif re.search(r'\bwarning(?:\([^)]*\))?:', line, re.IGNORECASE):
                warning_info = self._parse_warning_line(line)
                parsed['warning_details'].append(warning_info)
            elif any(keyword in line.lower() for keyword in ['compiling', 'building', 'linking']):
                parsed['compilation_info'].append(line)

        admitted = any('declaration uses `sorry`' in line for line in error_lines)
        compiled = result.get('returncode') == 0 and not parsed['error_details'] and not admitted

        # Parse source code directly if provided and no theorems/definitions found in output
        if compiled and source_code and (not parsed['theorems_proven'] and not parsed['definitions_created']):
            parsed.update(self._parse_source_code(source_code))

        # Extract theorems and definitions from source code if not found in output
        # Never count declarations from unrelated files in the working directory.
        if not compiled:
            for key in ('theorems_proven', 'lemmas_proven', 'definitions_created'):
                parsed[key] = []

        # Add verification status summary
        total_proofs = len(parsed['theorems_proven']) + len(parsed['lemmas_proven'])
        total_errors = len(parsed['error_details'])
        total_warnings = len(parsed['warning_details'])

        parsed['verification_status'] = {
            'total_proofs': total_proofs,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'success_rate': (total_proofs / (total_proofs + total_errors)) * 100 if (total_proofs + total_errors) > 0 else 100,
            'compilation_successful': compiled,
            'verification_complete': total_proofs > 0 and compiled,
            'compiler_returncode': result.get('returncode'),
            'has_admissions': admitted
        }

        # Add performance metrics
        if 'execution_time' in result:
            parsed['performance_metrics'].append({
                'metric': 'execution_time',
                'value': result['execution_time'],
                'unit': 'seconds'
            })

        # Add additional performance metrics even if parsing failed
        if parsed['verification_status']['compilation_successful']:
            parsed['performance_metrics'].extend([
                {
                    'metric': 'compilation_success',
                    'value': True,
                    'unit': 'boolean'
                },
                {
                    'metric': 'stdout_lines',
                    'value': len(stdout.split('\n')) if stdout else 0,
                    'unit': 'lines'
                },
                {
                    'metric': 'stderr_lines',
                    'value': len(stderr.split('\n')) if stderr else 0,
                    'unit': 'lines'
                }
            ])

        return parsed

    def _parse_source_code(self, source_code: str) -> Dict[str, Any]:
        """Inventory declarations outside comments and strings; this is not proof evidence."""
        from .lean_source import lean_code_without_comments
        source_code = lean_code_without_comments(source_code)
        parsed = {
            'theorems_proven': [],
            'lemmas_proven': [],
            'definitions_created': [],
            'axioms_defined': []
        }

        try:
            # Extract theorems
            theorem_matches = re.findall(r'\b(theorem|lemma|axiom)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:.]', source_code)
            for match_type, name in theorem_matches:
                if match_type == 'theorem':
                    parsed['theorems_proven'].append({
                        'type': 'theorem',
                        'name': name,
                        'source': 'direct_parsing',
                        'method': 'source_code_analysis'
                    })
                elif match_type == 'lemma':
                    parsed['lemmas_proven'].append({
                        'type': 'lemma',
                        'name': name,
                        'source': 'direct_parsing',
                        'method': 'source_code_analysis'
                    })
                elif match_type == 'axiom':
                    parsed['axioms_defined'].append({
                        'type': 'axiom',
                        'name': name,
                        'source': 'direct_parsing',
                        'method': 'source_code_analysis'
                    })

            # Extract definitions
            def_matches = re.findall(r'\b(def|definition|constant|abbrev)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:.]', source_code)
            for match_type, name in def_matches:
                parsed['definitions_created'].append({
                    'type': match_type,
                    'name': name,
                    'source': 'direct_parsing',
                    'method': 'source_code_analysis'
                })

            self.lean_logger.logger.info(f"Direct source parsing found: {len(parsed['theorems_proven'])} theorems, {len(parsed['definitions_created'])} definitions")

        except Exception as e:
            self.lean_logger.logger.warning(f"Direct source parsing failed: {e}")

        return parsed

    def _create_verification_files(self, proof_data: Dict[str, Any], verification_dir: Path,
                                 prefix: str, timestamp: str, saved_files: Dict[str, Path]):
        """Create verification files with available data"""
        try:
            verification_file = verification_dir / f"{prefix}_verification_{timestamp}.json"
            with open(verification_file, 'w') as f:
                json.dump({
                    'verification_status': proof_data.get('verification_status', {}),
                    'examples_verified': proof_data.get('examples_verified', []),
                    'computation_results': proof_data.get('computation_results', []),
                    'compilation_info': proof_data.get('compilation_info', [])
                }, f, indent=2, default=str)
            saved_files['verification'] = verification_file
        except Exception as e:
            self.lean_logger.logger.warning(f"Could not create verification file: {e}")

    def _create_performance_files(self, proof_data: Dict[str, Any], results: Dict[str, Any],
                                performance_dir: Path, prefix: str, timestamp: str, saved_files: Dict[str, Path]):
        """Create performance files with available metrics"""
        try:
            performance_file = performance_dir / f"{prefix}_performance_{timestamp}.json"
            with open(performance_file, 'w') as f:
                json.dump({
                    'performance_metrics': proof_data.get('performance_metrics', []),
                    'execution_time': results.get('execution_time', 0),
                    'stdout_length': len(results.get('stdout', '')),
                    'stderr_length': len(results.get('stderr', ''))
                }, f, indent=2, default=str)
            saved_files['performance'] = performance_file
        except Exception as e:
            self.lean_logger.logger.warning(f"Could not create performance file: {e}")

    def _parse_error_line(self, line: str) -> Dict[str, Any]:
        """Normalize compiler diagnostics while retaining their original text."""
        lower = line.lower()
        kind = ('syntax' if 'syntax error' in lower or 'unexpected token' in lower else
                'type_mismatch' if 'type mismatch' in lower else
                'type' if 'type error' in lower else 'unknown')
        return self._parse_diagnostic_location(line, 'error', kind)

    def _parse_warning_line(self, line: str) -> Dict[str, Any]:
        lower = line.lower()
        kind = ('unused_variable' if 'unused variable' in lower else
                'deprecated' if 'deprecated' in lower else 'unknown')
        diagnostic = self._parse_diagnostic_location(line, 'warning', kind)
        variable = re.search(r"unused variable ['`]([^'`]+)['`]", line)
        if variable:
            diagnostic['variable'] = variable.group(1)
        return diagnostic

    @staticmethod
    def _parse_diagnostic_location(raw: str, severity: str, kind: str) -> Dict[str, Any]:
        match = re.search(r'([^\s:]+\.lean):(\d+):(\d+)', raw)
        location = ({'file': match.group(1), 'line': int(match.group(2)),
                     'column': int(match.group(3))} if match else None)
        return {'raw_line': raw, 'message': raw, 'type': kind, 'severity': severity,
                'location': location, 'file': location['file'] if location else None,
                'line': location['line'] if location else None,
                'column': location['column'] if location else None}

    def run_theorem_verification(self, theorem_code: str, imports: List[str] = None) -> Dict[str, Any]:
        """Verify the submitted source and retain its kernel dependency evidence."""
        result = self.run_lean_code(theorem_code, imports)
        proofs = result.get('result', {})
        verified = proofs.get('theorems_proven', []) + proofs.get('lemmas_proven', [])
        result['verification_status'] = 'VERIFIED' if result['success'] and verified else 'FAILED'
        return result

    def run_algorithm_verification(self, algorithm_code: str, test_cases: List[Dict[str, Any]],
                                   imports: List[str] = None) -> Dict[str, Any]:
        """Compile an algorithm and execute each expected-value assertion with it."""
        compiled = self.run_lean_code(algorithm_code, imports)
        tests = []
        if compiled['success']:
            for index, case in enumerate(test_cases, 1):
                if 'expression' not in case or 'expected' not in case:
                    tests.append({'test_case': index, 'success': False, 'error': 'expression and expected are required'})
                    continue
                expected = str(case['expected']).lower() if isinstance(case['expected'], bool) else str(case['expected'])
                assertion = (f"\n#eval ((do\n  if ({case['expression']}) == ({expected}) then pure ()\n"
                             '  else throw (IO.userError "Algorithm expected-value assertion failed")) : IO Unit)\n')
                checked = self.run_lean_code(algorithm_code + assertion, imports)
                tests.append({'test_case': index, 'expression': case['expression'], 'expected': case['expected'],
                              'success': checked['success'], 'output': checked.get('stdout', ''),
                              'error': checked.get('error', '')})
        passed = sum(test['success'] for test in tests)
        success = compiled['success'] and passed == len(test_cases)
        return {'success': success, 'algorithm_verified': success, 'test_results': tests,
                'performance_metrics': {'total_tests': len(test_cases), 'passed_tests': passed,
                                        'failed_tests': len(test_cases) - passed,
                                        'success_rate': 100 * passed / len(test_cases) if test_cases else 0,
                                        'execution_time': compiled.get('execution_time', 0)}}

    def extract_mathematical_results(self, lean_output: str) -> Dict[str, Any]:
        """Extract mathematical results from Lean output"""
        results = {
            'theorems': [],
            'definitions': [],
            'computations': [],
            'proofs': []
        }

        # Prefer robust simple-name extraction to ensure we catch short artifact files
        # Extract theorems (name only)
        simple_theorems = re.findall(r"\btheorem\s+([A-Za-z0-9_]+)", lean_output)
        results['theorems'] = [{'name': name, 'statement': ''} for name in simple_theorems]

        # Extract definitions (name only)
        simple_defs = re.findall(r"\bdef\s+([A-Za-z0-9_]+)", lean_output)
        results['definitions'] = [{'name': name, 'body': ''} for name in simple_defs]

        # Extract computational results (#eval results)
        eval_pattern = r'#eval\s+(.*?)\s*-->\s*(.*?)(?=\n|$)'
        computations = re.findall(eval_pattern, lean_output)
        results['computations'] = [{'expression': c[0], 'result': c[1]} for c in computations]

        return results

    def create_mathematical_report(self, lean_code: str, results: Dict[str, Any]) -> str:
        """Create a comprehensive mathematical report"""
        report = f"""
# Mathematical Verification Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Original Code
```lean
{lean_code}
```

## Verification Results
"""

        if results.get('verification_status') == 'VERIFIED':
            report += "✅ **VERIFICATION SUCCESSFUL**\n\n"
        else:
            report += "❌ **VERIFICATION FAILED**\n\n"

        # Add performance metrics
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            report += ".1f"".2f"".3f"f"""
Success Rate: {metrics.get('success_rate', 0):.1f}%
Total Tests: {metrics.get('total_tests', 0)}
Passed: {metrics.get('passed_tests', 0)}
Failed: {metrics.get('failed_tests', 0)}
"""

        # Add extracted results
        extracted = self.extract_mathematical_results(results.get('stdout', ''))
        if extracted['theorems']:
            report += f"\n## Theorems Proven ({len(extracted['theorems'])})\n"
            for theorem in extracted['theorems']:
                report += f"- **{theorem['name']}**: {theorem['statement']}\n"

        if extracted['definitions']:
            report += f"\n## Definitions Created ({len(extracted['definitions'])})\n"
            for definition in extracted['definitions']:
                report += f"- **{definition['name']}**: {definition['body']}\n"

        if extracted['computations']:
            report += f"\n## Computational Results ({len(extracted['computations'])})\n"
            for comp in extracted['computations']:
                report += f"- `{comp['expression']}` → `{comp['result']}`\n"

        # Add errors if any
        if results.get('stderr'):
            report += f"\n## Errors and Warnings\n```\n{results['stderr']}\n```\n"

        return report

    def batch_verify_theorems(self, theorems: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Batch verify multiple theorems"""
        results = []

        for theorem_data in theorems:
            theorem_code = theorem_data.get('code', '')
            imports = theorem_data.get('imports', ["LeanNiche.Basic"])
            description = theorem_data.get('description', '')

            self.logger.info(f"Verifying theorem: {description}")

            result = self.run_theorem_verification(theorem_code, imports)
            result['description'] = description
            result['original_code'] = theorem_code

            results.append(result)

        return results

    def save_results(self, results: Dict[str, Any], filename: str,
                    format: str = 'json') -> Path:
        """Save results to file"""
        output_dir = Path("verification_results")
        output_dir.mkdir(exist_ok=True)

        filepath = output_dir / f"{filename}.{format}"

        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format == 'txt':
            with open(filepath, 'w') as f:
                f.write(self.create_mathematical_report(
                    results.get('original_code', ''),
                    results
                ))
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Results saved to: {filepath}")
        return filepath

    def save_comprehensive_proof_outcomes(self, results: Dict[str, Any],
                                        output_dir: Path, prefix: str = "proof_outcomes") -> Dict[str, Path]:
        """
        Save HONEST proof outcomes to multiple files.

        This method ONLY saves real verification results. It never generates fake data
        or pretends that proofs exist when they don't.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # Create subdirectories for organization.
        # If the caller already passed in the top-level `proofs` directory, avoid
        # creating a nested `proofs/proofs` structure. Use the provided directory
        # as the proof_dir in that case.
        if output_dir.name == 'proofs':
            proof_dir = output_dir
            verification_dir = output_dir / "verification"
            performance_dir = output_dir / "performance"
        else:
            proof_dir = output_dir / "proofs"
            verification_dir = output_dir / "verification"
            performance_dir = output_dir / "performance"

        for dir_path in [proof_dir, verification_dir, performance_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Get the actual proof data from results
        proof_data = results.get('result', {})

        # CRITICAL: Always create verification and performance files, regardless of compilation status
        verification_status = proof_data.get('verification_status', {})
        compilation_successful = verification_status.get('compilation_successful', False)

        # Always create verification and performance files with whatever data we have
        self._create_verification_files(proof_data, verification_dir, prefix, timestamp, saved_files)
        self._create_performance_files(proof_data, results, performance_dir, prefix, timestamp, saved_files)

        if not compilation_successful:
            # HONEST: If compilation failed, there are no verified theorems or definitions
            proof_data['theorems_proven'] = []
            proof_data['definitions_created'] = []
            proof_data['lemmas_proven'] = []
            proof_data['mathematical_properties'] = []
            self.logger.warning(f"Compilation failed for {prefix} - reporting zero verified proofs")

        # Save theorems and lemmas (ensure key always present even if empty)
        theorems_file = proof_dir / f"{prefix}_theorems_{timestamp}.json"
        with open(theorems_file, 'w') as f:
            json.dump({
                'theorems_proven': proof_data.get('theorems_proven', []),
                'lemmas_proven': proof_data.get('lemmas_proven', []),
                'axioms_defined': proof_data.get('axioms_defined', [])
            }, f, indent=2, default=str)
        saved_files['theorems'] = theorems_file

        # Save definitions and structures
        definitions_file = proof_dir / f"{prefix}_definitions_{timestamp}.json"
        with open(definitions_file, 'w') as f:
            json.dump({
                'definitions_created': proof_data.get('definitions_created', []),
                'structures_defined': proof_data.get('structures_defined', []),
                'inductive_types': proof_data.get('inductive_types', [])
            }, f, indent=2, default=str)
        saved_files['definitions'] = definitions_file

        # Save mathematical properties
        if proof_data.get('mathematical_properties'):
            properties_file = proof_dir / f"{prefix}_properties_{timestamp}.json"
            with open(properties_file, 'w') as f:
                json.dump({
                    'mathematical_properties': proof_data.get('mathematical_properties', []),
                    'tactics_used': proof_data.get('tactics_used', []),
                    'proof_methods': proof_data.get('proof_methods', [])
                }, f, indent=2, default=str)
            saved_files['properties'] = properties_file

        # Save error and warning details (only when compilation failed)
        if proof_data.get('error_details') or proof_data.get('warning_details'):
            issues_file = verification_dir / f"{prefix}_issues_{timestamp}.json"
            with open(issues_file, 'w') as f:
                json.dump({
                    'error_details': proof_data.get('error_details', []),
                    'warning_details': proof_data.get('warning_details', []),
                    'compilation_info': proof_data.get('compilation_info', [])
                }, f, indent=2, default=str)
            saved_files['issues'] = issues_file

        # Save complete comprehensive summary
        summary_file = output_dir / f"{prefix}_complete_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'execution_info': {
                    'timestamp': timestamp,
                    'prefix': prefix,
                    'success': results.get('success', False)
                },
                'complete_results': results,
                'saved_files': {k: str(v) for k, v in saved_files.items()}
            }, f, indent=2, default=str)
        saved_files['complete_summary'] = summary_file

        return saved_files

    def export_lean_code(self, code: str, output_path: Path) -> Path:
        """Export Lean code to a file.

        Args:
            code: The Lean code string to write.
            output_path: Destination Path where the .lean file will be written.

        Returns:
            Path: The path to the written Lean file.

        Notes:
            - Ensures parent directories exist.
            - Uses UTF-8 encoding to preserve symbols.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        self.logger.info(f"Exported Lean code to: {output_path}")
        return output_path

    # Backwards-compatibility wrapper name used by older examples
    def generate_proof_output(self, results: Dict[str, Any], output_dir: Path, prefix: str = "proof_outcomes") -> Dict[str, Path]:
        """Generate and save proof outputs and summaries.

        This method centralizes creation of JSON summaries and invokes the
        comprehensive saver to write structured proof artifacts.

        Args:
            results: The result dictionary returned by `run_lean_code` or similar.
            output_dir: Directory to write output files into.
            prefix: Filename prefix for generated artifacts.

        Returns:
            Dict[str, Path]: Mapping of artifact types to file paths created.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save a raw JSON summary first
        summary_json = output_dir / f"{prefix}_results.json"
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        # Use existing helper to save categorized proof outcomes
        saved = self.save_comprehensive_proof_outcomes(results, output_dir, prefix)
        saved['summary_json'] = summary_json

        # Consolidate any .lean artifacts into the saved JSONs so trivial theorems/defs
        # written directly to .lean files by orchestrators are captured in the JSON outputs.
        try:
            self._consolidate_lean_artifacts(saved, output_dir, prefix)
        except Exception as e:
            # Log but don't fail the generation
            self.logger.warning(f"Consolidation of lean artifacts failed: {e}")

        # No automatic insertion of artifact names here; CI should provide
        # a Lean environment where `LeanNiche` modules are discoverable so the
        # runner's parsing can capture declarations directly.

        self.logger.info(f"Generated proof outputs under: {output_dir}")
        return saved

    def _consolidate_lean_artifacts(self, saved_files: Dict[str, Path], output_dir: Path, prefix: str):
        """Record artifact sources without promoting them to verified proofs.

        Source files discovered in an output directory are unverified evidence.
        Reporting must never modify project sources or run dependency updates.
        """
        import hashlib

        if not output_dir.exists():
            return
        artifacts = []
        for path in sorted(output_dir.rglob("*.lean")):
            contents = path.read_bytes()
            artifacts.append({
                "path": str(path.relative_to(output_dir)),
                "sha256": hashlib.sha256(contents).hexdigest(),
                "verification_status": "unverified_source",
            })
        inventory = output_dir / f"{prefix}_source_inventory.json"
        inventory.write_text(json.dumps({"artifacts": artifacts}, indent=2) + "\n", encoding="utf-8")
        saved_files["source_inventory"] = inventory

    def suggest_atp_integration(self) -> str:
        """Return a short plan and resources for ATP/SMT integration with Lean.

        This helper does not perform integration but documents recommended
        approaches (Lean-auto, SMT tactics, delaborators) and test suggestions.

        Returns:
            str: Multi-line guidance text describing integration options and tests.
        """
        guidance = (
            "Recommended ATP/SMT integration approaches for Lean:\n"
            "1) Lean-auto / External ATPs: use a translation layer to export goals to ATPs\n"
            "   - Produce TPTP problems from Lean goals, call ATP (E, Vampire), and reconstruct proofs.\n"
            "   - Test by round-tripping small lemmas and asserting ATP-produced proof scripts typecheck.\n"
            "2) SMT solver tactics: integrate Lean-SMT style tactics to discharge decidable goals.\n"
            "   - Add a lightweight adapter that calls z3/cvc5 and returns witnesses.\n"
            "   - Write unit tests that ensure simple arithmetic and bitvector goals succeed.\n"
            "3) Delaborators & unexpanders: improve output formatting of proofs and definitions.\n"
            "   - Implement delaborators to format proof terms and use them when generating reports.\n"
            "4) I/O and proof export: use Lean `#print`, `#check`, and `#eval` to capture proofs and values.\n"
            "   - Export proof terms to files (.lean/.json) and include them in verification reports.\n"
            "5) Testing: create small, deterministic examples that exercise ATP/SMT flows and verify\n"
            "   outputs by checking that exported proof files `lean --make` or `lake` build successfully.\n"
            "Quick test checklist:\n"
            "- Round-trip: Lean goal -> ATP -> reconstructed proof -> Lean typechecks.\n"
            "- Timeout handling: ensure timeouts produce clear diagnostics recorded in `error_details`.\n"
            "- Deterministic seeds: tests should be repeatable (use fixed random seeds where needed).\n"
        )
        return guidance


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Lean Code Runner and Verifier")
    parser.add_argument('--code', help='Lean code to execute')
    parser.add_argument('--file', help='Lean file to execute')
    parser.add_argument('--imports', nargs='*', default=['LeanNiche.Basic'],
                       help='Imports to include')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--format', choices=['json', 'txt'], default='json',
                       help='Output format')

    args = parser.parse_args()

    runner = LeanRunner()

    if args.code:
        results = runner.run_lean_code(args.code, args.imports)
    elif args.file:
        with open(args.file, 'r') as f:
            code = f.read()
        results = runner.run_lean_code(code, args.imports)
    else:
        print("Error: Please provide either --code or --file")
        return 1

    if args.output:
        runner.save_results(results, args.output, args.format)
    else:
        print(json.dumps(results, indent=2, default=str))

    return 0 if results.get('success', False) else 1


if __name__ == "__main__":
    exit(main())
