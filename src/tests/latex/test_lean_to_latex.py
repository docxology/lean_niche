"""Tests for Lean to LaTeX conversion module"""

import pytest
import tempfile
import os
from pathlib import Path
from src.latex.lean_to_latex import LeanToLatexConverter


class TestLeanToLatexConverter:
    """Test LeanToLatexConverter class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.converter = LeanToLatexConverter()

    def test_initialization(self):
        """Test converter initialization"""
        assert isinstance(self.converter.symbol_map, dict)
        assert isinstance(self.converter.environment_map, dict)
        assert len(self.converter.symbol_map) > 0
        assert len(self.converter.environment_map) > 0

    def test_convert_symbol(self):
        """Test symbol conversion"""
        # Test basic symbols
        assert self.converter.convert_symbol('∧') == r'\land'
        assert self.converter.convert_symbol('→') == r'\rightarrow'
        assert self.converter.convert_symbol('∀') == r'\forall'
        assert self.converter.convert_symbol('∈') == r'\in'
        assert self.converter.convert_symbol('⊆') == r'\subseteq'
        assert self.converter.convert_symbol('≤') == r'\leq'
        assert self.converter.convert_symbol('≠') == r'\neq'

        # Test Greek letters
        assert self.converter.convert_symbol('α') == r'\alpha'
        assert self.converter.convert_symbol('β') == r'\beta'
        assert self.converter.convert_symbol('γ') == r'\gamma'
        assert self.converter.convert_symbol('λ') == r'\lambda'
        assert self.converter.convert_symbol('π') == r'\pi'

        # Test blackboard bold
        assert self.converter.convert_symbol('ℕ') == r'\mathbb{N}'
        assert self.converter.convert_symbol('ℤ') == r'\mathbb{Z}'
        assert self.converter.convert_symbol('ℚ') == r'\mathbb{Q}'
        assert self.converter.convert_symbol('ℝ') == r'\mathbb{R}'
        assert self.converter.convert_symbol('ℂ') == r'\mathbb{C}'

        # Test unknown symbols (should return unchanged)
        assert self.converter.convert_symbol('unknown') == 'unknown'
        assert self.converter.convert_symbol('x') == 'x'

    def test_convert_expression(self):
        """Test expression conversion"""
        # Test simple expression
        expr = "∀ x ∈ ℝ, x ≥ 0"
        result = self.converter.convert_expression(expr)
        expected = r"\forall x \in \mathbb{R}, x \geq 0"
        assert result == expected

        # Test function application
        expr = "f(x) = x² + 2*x + 1"
        result = self.converter.convert_expression(expr)
        expected = r"\text{f}(x) = x² + 2*x + 1"
        assert result == expected

        # Test type annotations
        expr = "x : ℝ → y : ℝ"
        result = self.converter.convert_expression(expr)
        expected = r"x : \mathbb{R} \rightarrow y : \mathbb{R}"
        assert result == expected

        # Test complex expression
        expr = "∃ ε > 0, ∀ δ > 0, |x - a| < δ → |f(x) - L| < ε"
        result = self.converter.convert_expression(expr)
        expected = r"\exists \epsilon > 0, \forall \delta > 0, |x - a| < \delta \rightarrow |\text{f}(x) - L| < \epsilon"
        assert result == expected

    def test_convert_theorem(self):
        """Test theorem conversion"""
        # Test simple theorem
        lean_theorem = "theorem add_comm : ∀ a b : ℕ, a + b = b + a"
        result = self.converter.convert_theorem(lean_theorem)

        assert r"\begin{theorem}{add_comm}" in result
        assert r"\label{thm:add_comm}" in result
        assert r"\forall a b : \mathbb{N}, a + b = b + a" in result
        assert r"\end{theorem}" in result

        # Test lemma
        lean_lemma = "lemma mul_comm : ∀ a b : ℕ, a * b = b * a"
        result = self.converter.convert_theorem(lean_lemma)

        assert r"\begin{lemma}{mul_comm}" in result
        assert r"\label{thm:mul_comm}" in result
        assert r"\end{lemma}" in result

        # Test invalid theorem (should fall back to expression conversion)
        invalid_theorem = "invalid theorem format"
        result = self.converter.convert_theorem(invalid_theorem)
        assert result == "invalid theorem format"

    def test_convert_definition(self):
        """Test definition conversion"""
        # Test simple definition
        lean_def = "def factorial : ℕ → ℕ | 0 => 1 | n + 1 => (n + 1) * factorial n"
        result = self.converter.convert_definition(lean_def)

        assert r"\begin{definition}{factorial}" in result
        assert r"\label{def:factorial}" in result
        assert r"\end{definition}" in result

        # Test definition with complex body
        lean_def = "definition fibonacci : ℕ → ℕ := λ n => match n with | 0 => 0 | 1 => 1 | k + 2 => fibonacci k + fibonacci (k + 1)"
        result = self.converter.convert_definition(lean_def)

        assert r"\begin{definition}{fibonacci}" in result
        assert r"\label{def:fibonacci}" in result
        assert r"\end{definition}" in result

        # Test invalid definition (should fall back to expression conversion)
        invalid_def = "invalid definition format"
        result = self.converter.convert_definition(invalid_def)
        assert result == "invalid definition format"

    def test_convert_namespace(self):
        """Test namespace conversion"""
        # Test valid namespace
        lean_ns = "namespace LeanNiche.Basic"
        result = self.converter.convert_namespace(lean_ns)
        expected = r"\section{LeanNiche.Basic}\label{sec:LeanNiche.Basic}"
        assert result == expected

        # Test invalid namespace (should return unchanged)
        invalid_ns = "invalid namespace format"
        result = self.converter.convert_namespace(invalid_ns)
        assert result == invalid_ns

    def test_convert_file(self):
        """Test complete file conversion"""
        # Create a temporary Lean file
        lean_content = """-- This is a test Lean file
namespace TestModule

/-- Addition is commutative -/
theorem add_comm : ∀ a b : ℕ, a + b = b + a :=
  sorry

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

end TestModule"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
            f.write(lean_content)
            temp_file = Path(f.name)

        try:
            result = self.converter.convert_file(temp_file)

            # Check LaTeX document structure
            assert r"\documentclass{article}" in result
            assert r"\usepackage{amssymb,amsmath,amsthm}" in result
            assert r"\begin{document}" in result
            assert r"\end{document}" in result

            # Check content conversion
            assert r"\section{TestModule}" in result
            assert r"\begin{theorem}{add_comm}" in result
            assert r"\begin{definition}{factorial}" in result

        finally:
            # Clean up
            os.unlink(temp_file)

    def test_convert_file_with_output(self):
        """Test file conversion with output file"""
        lean_content = """namespace Test
theorem test_thm : ∀ x : ℝ, x = x
end Test"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
            f.write(lean_content)
            input_file = Path(f.name)

        output_file = Path(f"{input_file}.tex")

        try:
            result = self.converter.convert_file(input_file, output_file)

            # Check that output file was created
            assert output_file.exists()

            # Read the output file and verify content
            with open(output_file, 'r') as f:
                output_content = f.read()

            assert r"\documentclass{article}" in output_content
            assert r"\section{Test}" in output_content
            assert r"\begin{theorem}{test_thm}" in output_content

        finally:
            # Clean up
            os.unlink(input_file)
            if output_file.exists():
                os.unlink(output_file)

    def test_convert_file_not_found(self):
        """Test file conversion with non-existent file"""
        with pytest.raises(FileNotFoundError):
            self.converter.convert_file(Path("/non/existent/file.lean"))

    def test_process_block(self):
        """Test block processing"""
        # Test theorem block
        theorem_block = "theorem test : ∀ x, x = x"
        result = self.converter.process_block(theorem_block)
        assert r"\begin{theorem}{test}" in result

        # Test definition block
        def_block = "def test_def : ℕ := 42"
        result = self.converter.process_block(def_block)
        assert r"\begin{definition}{test_def}" in result

        # Test regular expression block
        expr_block = "∀ x ∈ ℝ, x ≥ 0"
        result = self.converter.process_block(expr_block)
        assert result == r"\forall x \in \mathbb{R}, x \geq 0"

    def test_symbol_map_completeness(self):
        """Test that symbol map contains expected mathematical symbols"""
        expected_symbols = [
            '∧', '∨', '¬', '→', '↔', '∀', '∃', '∈', '⊆', '⊂',
            '∩', '∪', '∖', '×', '⊥', '⊤', '∅', '≤', '≥', '≠',
            'α', 'β', 'γ', 'λ', 'π', 'ℕ', 'ℤ', 'ℚ', 'ℝ', 'ℂ'
        ]

        for symbol in expected_symbols:
            assert symbol in self.converter.symbol_map, f"Missing symbol: {symbol}"

    def test_environment_map_completeness(self):
        """Test that environment map contains expected Lean environments"""
        expected_envs = ['theorem', 'lemma', 'definition', 'def', 'example', 'namespace']

        for env in expected_envs:
            assert env in self.converter.environment_map, f"Missing environment: {env}"
