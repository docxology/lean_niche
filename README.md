# LeanNiche - Deep Research Environment

A comprehensive Lean 4 research environment for deep mathematical proofs, algorithm verification, and formal methods research.

## 🚀 Features

### Core Capabilities
- **Mathematical Proofs**: From basic arithmetic to advanced number theory
- **Algorithm Verification**: Verified implementations of sorting, searching, and computational algorithms
- **Set Theory & Logic**: Complete formalization of set operations and logical reasoning
- **Automated Tactics**: Demonstrates Lean's powerful proof automation
- **Computational Mathematics**: Verified numerical algorithms and sequences
- **Test Suite**: Comprehensive testing framework for all components

### Project Structure
```
lean_niche/
├── src/                        # Source code
│   ├── LeanNiche/
│   │   ├── Basic.lean         # Fundamental mathematical proofs
│   │   ├── Advanced.lean      # Advanced theorems and proofs
│   │   ├── Tactics.lean       # Proof automation techniques
│   │   ├── SetTheory.lean     # Set theory and logic
│   │   └── Computational.lean # Algorithms and computation
│   └── Main.lean              # Main executable
├── tests/
│   └── TestSuite.lean         # Comprehensive test suite
├── docs/                      # Documentation
├── examples/                  # Example usage
├── scripts/                   # Build and utility scripts
├── lakefile.toml              # Project configuration
└── lean-toolchain             # Lean version specification
```

## 🛠 Installation & Setup

### Prerequisites
- **Lean 4**: Install via Elan (Lean version manager)
- **Git**: For dependency management
- **VS Code** (recommended): With Lean 4 extension

### Quick Setup
1. **Install Elan** (if not already installed):
   ```bash
   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd lean_niche
   lake update  # Download Mathlib4
   ```

3. **Build the project**:
   ```bash
   lake build
   ```

4. **Run the environment**:
   ```bash
   lake exe lean_niche
   ```

## 📚 Core Modules

### Basic.lean
Fundamental mathematical proofs demonstrating:
- Natural number arithmetic (commutativity, associativity)
- Basic algebraic properties
- Simple induction proofs

**Key Theorems**:
- `add_comm`: Addition is commutative
- `add_assoc`: Addition is associative
- `mul_add`: Multiplication distributes over addition

### Advanced.lean
Advanced mathematical concepts including:
- Prime number theory
- Number sequences (Fibonacci, factorials)
- Infinite descent proofs

**Key Theorems**:
- `prime_gt_two_odd`: Primes > 2 are odd
- `infinite_primes`: There are infinitely many primes
- `factorial_strict_mono`: Factorial is strictly increasing

### Tactics.lean
Demonstrates Lean's proof automation:
- Basic tactics (`rw`, `simp`, `induction`)
- Automated reasoning (`linarith`, `omega`)
- Proof by computation (`norm_num`)
- Existential and universal proofs

### SetTheory.lean
Complete set theory formalization:
- Set operations (union, intersection, complement)
- Set relations (subset, membership)
- Functions between sets
- Logical equivalence and quantifiers

### Computational.lean
Verified computational algorithms:
- Fibonacci sequence computation
- Insertion sort with correctness proof
- Factorial and power functions
- Greatest common divisor algorithm

## 🧪 Testing

The comprehensive test suite includes:
- **Unit tests** for all functions
- **Property tests** for mathematical theorems
- **Integration tests** across modules
- **Performance benchmarks**

Run tests:
```bash
lake exe lean_niche  # Runs integrated test suite
```

## 📖 Usage Examples

### Basic Proof
```lean
import LeanNiche.Basic

-- Use basic arithmetic theorems
example (a b : ℕ) : a + b = b + a := by
  exact LeanNiche.Basic.add_comm a b
```

### Advanced Theorem
```lean
import LeanNiche.Advanced

-- Use advanced number theory
example : ∀ n : ℕ, n > 1 → ∃ p : ℕ, p.Prime ∧ p ≤ n! + 1 := by
  intro n hn
  -- Proof using advanced techniques
```

### Algorithm Verification
```lean
import LeanNiche.Computational

-- Use verified algorithms
def sorted_list := LeanNiche.Computational.insertion_sort [3,1,4,1,5]
-- Guaranteed to be sorted by construction
```

## 🔧 Build System

### Available Commands
- `lake build`: Build all modules
- `lake exe lean_niche`: Run main executable
- `lake clean`: Clean build artifacts
- `lake update`: Update dependencies

### Custom Scripts
Located in `scripts/` directory:
- `build.sh`: Optimized build script
- `test.sh`: Run all tests
- `analyze.sh`: Code analysis and metrics
- `docs.sh`: Generate documentation

## 📊 Research Applications

### Areas of Focus
1. **Formal Mathematics**: Rigorous proofs of mathematical theorems
2. **Algorithm Verification**: Proving correctness of computational methods
3. **Logic and Foundations**: Exploring mathematical logic and type theory
4. **Automated Reasoning**: Developing new proof tactics and automation
5. **Computational Number Theory**: Verified implementations of number-theoretic algorithms

### Research Workflow
1. **Explore**: Use existing theorems and proofs as starting points
2. **Develop**: Add new theorems and proofs to relevant modules
3. **Verify**: Use the test suite to ensure correctness
4. **Document**: Update documentation with new findings
5. **Share**: Contribute improvements back to the community

## 🎯 Learning Path

### Beginner
1. Study `Basic.lean` - Learn fundamental proof techniques
2. Complete exercises in `examples/`
3. Run and understand the test suite

### Intermediate
1. Explore `Tactics.lean` - Master proof automation
2. Study `SetTheory.lean` - Understand formal logic
3. Implement your own verified algorithms

### Advanced
1. Work with `Advanced.lean` - Tackle complex theorems
2. Contribute to `Computational.lean` - Add new verified algorithms
3. Extend the test framework

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make changes following the established patterns
4. Add tests for new functionality
5. Submit a pull request

### Code Standards
- Follow Lean's naming conventions
- Add comprehensive documentation
- Include proofs for all theorems
- Test all new functionality
- Update README for significant changes

## 📄 License

This project is open source. See LICENSE file for details.

## 🔗 Resources

- [Lean Official Website](https://lean-lang.org/)
- [Mathlib4 Documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [Lean Zulip Chat](https://leanprover.zulipchat.com/)
- [Theorem Proving in Lean 4](https://leanprover.github.io/theorem_proving_in_lean4/)

## 📈 Performance

The environment is optimized for:
- **Fast compilation** of large proof developments
- **Efficient proof checking** with minimal overhead
- **Scalable dependency management** via Lake
- **Incremental builds** for rapid development cycles

---

**LeanNiche**: Where formal mathematics meets computational verification.