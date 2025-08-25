#!/usr/bin/env python3
"""
Lean to LaTeX Converter
Converts Lean mathematical definitions and theorems to LaTeX format.

This module provides functionality to translate Lean mathematical content
into LaTeX equations and environments for documentation and publication.
"""

import re
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class LeanToLatexConverter:
    """Converter for Lean mathematical content to LaTeX"""

    def __init__(self):
        # Lean to LaTeX symbol mappings
        self.symbol_map = {
            # Logical operators
            '∧': r'\land',
            '∨': r'\lor',
            '¬': r'\neg',
            '→': r'\rightarrow',
            '↔': r'\leftrightarrow',
            '∀': r'\forall',
            '∃': r'\exists',
            '∃!': r'\exists!',
            '≠': r'\neq',

            # Set theory
            '∈': r'\in',
            '∉': r'\notin',
            '⊆': r'\subseteq',
            '⊂': r'\subset',
            '⊇': r'\supseteq',
            '⊃': r'\supset',
            '∩': r'\cap',
            '∪': r'\cup',
            '∖': r'\setminus',
            '×': r'\times',
            '⊥': r'\bot',
            '⊤': r'\top',
            '∅': r'\emptyset',
            '𝒫': r'\mathcal{P}',
            '℘': r'\wp',

            # Relations
            '≤': r'\leq',
            '≥': r'\geq',
            '<': r'<',
            '>': r'>',
            '≡': r'\equiv',
            '≈': r'\approx',
            '∼': r'\sim',
            '≪': r'\ll',
            '≫': r'\gg',
            '≍': r'\asymp',
            '∝': r'\propto',

            # Arithmetic
            '÷': r'\div',
            '⋅': r'\cdot',
            '×': r'\times',
            '√': r'\sqrt',
            '∑': r'\sum',
            '∏': r'\prod',
            '∫': r'\int',
            '∂': r'\partial',
            '∇': r'\nabla',

            # Greek letters commonly used in Lean
            'α': r'\alpha',
            'β': r'\beta',
            'γ': r'\gamma',
            'δ': r'\delta',
            'ε': r'\epsilon',
            'ζ': r'\zeta',
            'η': r'\eta',
            'θ': r'\theta',
            'ι': r'\iota',
            'κ': r'\kappa',
            'λ': r'\lambda',
            'μ': r'\mu',
            'ν': r'\nu',
            'ξ': r'\xi',
            'π': r'\pi',
            'ρ': r'\rho',
            'σ': r'\sigma',
            'τ': r'\tau',
            'υ': r'\upsilon',
            'φ': r'\phi',
            'χ': r'\chi',
            'ψ': r'\psi',
            'ω': r'\omega',
            'Γ': r'\Gamma',
            'Δ': r'\Delta',
            'Θ': r'\Theta',
            'Λ': r'\Lambda',
            'Π': r'\Pi',
            'Σ': r'\Sigma',
            'Φ': r'\Phi',
            'Ψ': r'\Psi',
            'Ω': r'\Omega',

            # Blackboard bold
            'ℕ': r'\mathbb{N}',
            'ℤ': r'\mathbb{Z}',
            'ℚ': r'\mathbb{Q}',
            'ℝ': r'\mathbb{R}',
            'ℂ': r'\mathbb{C}',

            # Calligraphic
            '𝒩': r'\mathcal{N}',
            'ℳ': r'\mathcal{M}',
            'ℛ': r'\mathcal{R}',
            '𝒜': r'\mathcal{A}',
            'ℬ': r'\mathcal{B}',
            '𝒞': r'\mathcal{C}',
            '𝒟': r'\mathcal{D}',
            'ℰ': r'\mathcal{E}',
            'ℱ': r'\mathcal{F}',
            '𝒢': r'\mathcal{G}',
            'ℋ': r'\mathcal{H}',
            'ℐ': r'\mathcal{I}',
            '𝒥': r'\mathcal{J}',
            '𝒦': r'\mathcal{K}',
            'ℒ': r'\mathcal{L}',
        }

        # Lean environment mappings
        self.environment_map = {
            'theorem': 'theorem',
            'lemma': 'lemma',
            'definition': 'definition',
            'def': 'definition',
            'example': 'example',
            'namespace': 'section',
            'section': 'subsection',
            'structure': 'definition',
            'inductive': 'definition',
        }

    def convert_symbol(self, symbol: str) -> str:
        """Convert a single Lean symbol to LaTeX"""
        return self.symbol_map.get(symbol, symbol)

    def convert_expression(self, expr: str) -> str:
        """Convert a Lean mathematical expression to LaTeX"""
        # Replace symbols
        result = expr
        for lean_symbol, latex_symbol in self.symbol_map.items():
            result = result.replace(lean_symbol, latex_symbol)

        # Handle function applications
        result = re.sub(r'(\w+)\s*\(', r'\\text{\1}(', result)

        # Handle type annotations
        result = re.sub(r':\s*(\w+)', r' : \1', result)

        return result

    def convert_theorem(self, lean_code: str) -> str:
        """Convert a Lean theorem to LaTeX"""
        # Extract theorem name and statement
        theorem_match = re.match(r'(theorem|lemma)\s+(\w+)\s*:(.*)', lean_code.strip())
        if not theorem_match:
            return self.convert_expression(lean_code)

        env_type, name, statement = theorem_match.groups()
        latex_env = self.environment_map.get(env_type, env_type)

        # Convert the statement
        latex_statement = self.convert_expression(statement)

        # Generate LaTeX
        latex = f"\\begin{{{latex_env}}}{{{name}}}\n"
        latex += f"\\label{{thm:{name}}}\n"
        latex += f"{latex_statement}\n"
        latex += f"\\end{{{latex_env}}}"

        return latex

    def convert_definition(self, lean_code: str) -> str:
        """Convert a Lean definition to LaTeX"""
        # Extract definition name and body
        def_match = re.match(r'(def|definition)\s+(\w+).*:(.*)', lean_code.strip())
        if not def_match:
            return self.convert_expression(lean_code)

        env_type, name, body = def_match.groups()
        latex_env = self.environment_map.get(env_type, 'definition')

        # Convert the body
        latex_body = self.convert_expression(body)

        # Generate LaTeX
        latex = f"\\begin{{{latex_env}}}{{{name}}}\n"
        latex += f"\\label{{def:{name}}}\n"
        latex += f"{latex_body}\n"
        latex += f"\\end{{{latex_env}}}"

        return latex

    def convert_namespace(self, lean_code: str) -> str:
        """Convert a Lean namespace to LaTeX section"""
        ns_match = re.match(r'namespace\s+([\w\.]+)', lean_code.strip())
        if not ns_match:
            return lean_code

        name = ns_match.group(1)
        # Preserve full namespace with dots (e.g., LeanNiche.Basic)
        return f"\\section{{{name}}}\\label{{sec:{name}}}"

    def convert_file(self, lean_file: Path, output_file: Optional[Path] = None) -> str:
        """Convert a complete Lean file to LaTeX"""
        if not lean_file.exists():
            raise FileNotFoundError(f"Lean file not found: {lean_file}")

        with open(lean_file, 'r', encoding='utf-8') as f:
            lean_content = f.read()

        # Split into lines and process each
        lines = lean_content.split('\n')
        latex_lines = []
        in_theorem = False
        in_definition = False
        current_block = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('--'):
                continue

            # Handle different Lean constructs
            if line.startswith('namespace'):
                latex_lines.append(self.convert_namespace(line))
            elif line.startswith(('theorem', 'lemma')):
                if current_block:
                    latex_lines.append(self.process_block('\n'.join(current_block)))
                current_block = [line]
                in_theorem = True
            elif line.startswith(('def', 'definition')):
                if current_block:
                    latex_lines.append(self.process_block('\n'.join(current_block)))
                current_block = [line]
                in_definition = True
            elif line.startswith('end'):
                current_block.append(line)
                latex_lines.append(self.process_block('\n'.join(current_block)))
                current_block = []
                in_theorem = False
                in_definition = False
            elif in_theorem or in_definition:
                current_block.append(line)
            else:
                # Regular mathematical expression
                latex_lines.append(self.convert_expression(line))

        # Process any remaining block
        if current_block:
            latex_lines.append(self.process_block('\n'.join(current_block)))

        # Combine all LaTeX
        latex_content = '\n'.join(latex_lines)

        # Add LaTeX document structure
        full_latex = f"""\\documentclass{{article}}
\\usepackage{{amssymb,amsmath,amsthm}}
\\usepackage{{hyperref}}
\\usepackage{{cleveref}}

\\newtheorem{{theorem}}{{Theorem}}[section]
\\newtheorem{{lemma}}[theorem]{{Lemma}}
\\newtheorem{{definition}}[theorem]{{Definition}}
\\newtheorem{{example}}[theorem]{{Example}}

\\title{{LeanNiche Mathematical Content}}
\\author{{LeanNiche Team}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle
\\tableofcontents

{latex_content}

\\end{{document}}"""

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_latex)

        return full_latex

    def process_block(self, block: str) -> str:
        """Process a block of Lean code"""
        if block.startswith(('theorem', 'lemma')):
            return self.convert_theorem(block)
        elif block.startswith(('def', 'definition')):
            return self.convert_definition(block)
        else:
            return self.convert_expression(block)

def main():
    """Main function for command-line usage"""
    if len(sys.argv) != 3:
        print("Usage: python lean_to_latex.py <input.lean> <output.tex>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    converter = LeanToLatexConverter()
    try:
        latex = converter.convert_file(input_file, output_file)
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error converting file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
