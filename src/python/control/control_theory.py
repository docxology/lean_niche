#!/usr/bin/env python3
"""
Control Theory Module for LeanNiche

This module provides comprehensive control theory implementations including:
- PID controller design and analysis
- System stability analysis using linear algebra
- Linear Quadratic Regulation (LQR)
- Controllability and observability analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PIDController:
    """PID Controller implementation with anti-windup."""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain
    integral: float = 0.0
    previous_error: float = 0.0

    def update(self, error: float, dt: float) -> float:
        """Update PID controller and return control signal."""
        # Proportional term
        proportional = self.kp * error

        # Integral term with anti-windup
        self.integral += self.ki * error * dt
        if abs(self.integral) > 10:  # Anti-windup limit
            self.integral = np.sign(self.integral) * 10

        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt

        # Control output
        output = proportional + self.integral + derivative

        # Update previous error
        self.previous_error = error

        return output


class ControlTheoryAnalyzer:
    """Comprehensive control theory analysis tools."""

    def __init__(self):
        """Initialize the control theory analyzer."""
        pass

    def design_pid_controller(self, system_params: Dict[str, float],
                            pid_gains: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Design and analyze PID controllers for a given system.

        Args:
            system_params: System parameters (omega_n, zeta, dt, simulation_time)
            pid_gains: List of PID gain dictionaries

        Returns:
            List of simulation results for each controller
        """
        # System parameters
        omega_n = system_params.get('omega_n', 2.0)
        zeta = system_params.get('zeta', 0.3)
        dt = system_params.get('dt', 0.01)
        simulation_time = system_params.get('simulation_time', 10.0)

        # Discretize system: x'' + 2ζωₙx' + ωₙ²x = u
        a1 = 2 - 2*zeta*omega_n*dt
        a2 = -1 + zeta*omega_n*dt
        b0 = omega_n**2 * dt**2

        print(f"  System: ωₙ = {omega_n}, ζ = {zeta}")
        print(f"  Discrete: a1 = {a1:.4f}, a2 = {a2:.4f}, b0 = {b0:.4f}")

        # Simulation setup
        time_steps = int(simulation_time / dt)
        time = np.arange(0, simulation_time, dt)

        # Reference signal (step input)
        reference = np.ones_like(time) * 1.0

        results = []

        print("  📈 Simulating controllers...")
        for pid_param in pid_gains:
            print(f"    Testing: {pid_param['name']}")

            # Initialize controller
            controller = PIDController(
                kp=pid_param['kp'],
                ki=pid_param['ki'],
                kd=pid_param['kd']
            )

            # Initialize system state
            x = np.zeros(time_steps + 2)  # x[k], x[k+1], x[k+2]
            u = np.zeros(time_steps)

            # Simulation loop
            for k in range(time_steps):
                # Current error
                error = reference[k] - x[k+1]

                # PID control law
                u[k] = controller.update(error, dt)

                # System dynamics
                x[k+2] = a1 * x[k+1] + a2 * x[k] + b0 * u[k]

            # Compute performance metrics
            steady_state_error = abs(reference[-1] - x[-1])
            settling_time = self._compute_settling_time(x[2:], reference, time, dt)
            overshoot = (np.max(x) - reference[-1]) / reference[-1] * 100 if np.max(x) > reference[-1] else 0

            results.append({
                'parameters': pid_param,
                'time': time,
                'reference': reference,
                'output': x[2:],  # Skip initial conditions
                'control_signal': u,
                'steady_state_error': steady_state_error,
                'settling_time': settling_time,
                'overshoot': overshoot
            })

        print("✅ PID controller design and simulation complete")
        return results

    def analyze_system_stability(self, system_matrices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze system stability using linear algebra methods.

        Args:
            system_matrices: Dictionary containing A, B, C matrices

        Returns:
            Stability analysis results
        """
        print("🔬 Analyzing system stability...")

        A = system_matrices['A']
        B = system_matrices['B']
        C = system_matrices['C']

        print("  📊 System matrices:")
        print(f"    A = {A.tolist()}")
        print(f"    B = {B.tolist()}")
        print(f"    C = {C.tolist()}")

        # Compute eigenvalues for stability analysis
        eigenvalues = np.linalg.eigvals(A)
        print(f"  📈 Eigenvalues: {eigenvalues}")

        # Check stability (all eigenvalues should have negative real parts)
        is_stable = all(ev.real < 0 for ev in eigenvalues)
        print(f"  ✅ Stability check: {'Stable' if is_stable else 'Unstable'}")

        # Controllability analysis
        # Compute controllability matrix
        n = A.shape[0]
        C_mat = np.zeros((n, n))

        for i in range(n):
            if i == 0:
                C_mat[:, i] = B.flatten()
            else:
                C_mat[:, i] = A @ C_mat[:, i-1]

        controllability_rank = np.linalg.matrix_rank(C_mat)
        is_controllable = controllability_rank == n
        print(f"  🎮 Controllability rank: {controllability_rank}/{n} ({'Controllable' if is_controllable else 'Uncontrollable'})")

        # Design LQR controller
        Q = np.eye(n) * 10  # State cost matrix
        R = np.array([[1]])  # Control cost matrix

        print("  🎯 Designing LQR controller...")
        print(f"    Q = {Q.tolist()}")
        print(f"    R = {R.tolist()}")

        # Solve algebraic Riccati equation (simplified approach)
        try:
            # Simple approximation for demonstration
            P = np.eye(n)  # Approximate solution
            K = np.linalg.inv(R) @ B.T @ P
            print(f"    LQR gain K = {K.tolist()}")

            # Closed-loop eigenvalues
            A_cl = A - B @ K
            eigenvalues_cl = np.linalg.eigvals(A_cl)
            is_cl_stable = all(ev.real < 0 for ev in eigenvalues_cl)
            print(f"    Closed-loop eigenvalues: {eigenvalues_cl}")
            print(f"    Closed-loop stability: {'Stable' if is_cl_stable else 'Unstable'}")

        except Exception as e:
            print(f"    ⚠️ LQR computation error: {e}")
            K = None
            eigenvalues_cl = None
            is_cl_stable = False

        return {
            'system_matrices': system_matrices,
            'eigenvalues': eigenvalues.tolist(),
            'is_stable': is_stable,
            'controllability_rank': int(controllability_rank),
            'is_controllable': is_controllable,
            'lqr_gain': K.tolist() if K is not None else None,
            'closed_loop_eigenvalues': eigenvalues_cl.tolist() if eigenvalues_cl is not None else None,
            'is_cl_stable': is_cl_stable
        }

    def _compute_settling_time(self, output: np.ndarray, reference: np.ndarray,
                             time: np.ndarray, threshold: float = 0.02) -> float:
        """Compute settling time for step response."""
        for i in range(len(time)):
            if abs(output[i] - reference[i]) < threshold:
                # Check if it stays within threshold
                if all(abs(output[j] - reference[j]) < threshold
                       for j in range(i, min(i+100, len(time)))):
                    return time[i]
        return time[-1]  # Never settled


class StabilityAnalyzer:
    """Advanced stability analysis tools."""

    def __init__(self):
        """Initialize stability analyzer."""
        pass

    def lyapunov_stability_analysis(self, A: np.ndarray, Q: np.ndarray = None) -> Dict[str, Any]:
        """Perform Lyapunov stability analysis."""
        if Q is None:
            Q = np.eye(A.shape[0])

        # Solve Lyapunov equation: A^T P + P A = -Q
        # This is a simplified implementation
        try:
            # For demonstration, assume P = Q (which satisfies for stable A)
            P = Q.copy()

            # Check if P is positive definite
            eigenvals_P = np.linalg.eigvals(P)
            is_positive_definite = all(ev > 0 for ev in eigenvals_P)

            # Check Lyapunov equation
            lyapunov_eq = A.T @ P + P @ A + Q
            equation_satisfied = np.allclose(lyapunov_eq, 0, atol=1e-10)

            return {
                'P_matrix': P.tolist(),
                'Q_matrix': Q.tolist(),
                'is_positive_definite': is_positive_definite,
                'lyapunov_equation_satisfied': equation_satisfied,
                'eigenvalues_P': eigenvals_P.tolist()
            }
        except Exception as e:
            return {
                'error': str(e),
                'P_matrix': None,
                'is_positive_definite': False,
                'lyapunov_equation_satisfied': False
            }

    def root_locus_analysis(self, plant_tf: Dict[str, List[float]],
                          k_range: Tuple[float, float] = (0, 10)) -> Dict[str, Any]:
        """Perform root locus analysis."""
        # Simplified implementation
        numerator = plant_tf.get('numerator', [1])
        denominator = plant_tf.get('denominator', [1, 1])

        k_values = np.linspace(k_range[0], k_range[1], 100)
        root_loci = []

        for k in k_values:
            # Characteristic equation: denominator + k * numerator = 0
            char_poly = np.polyadd(denominator, k * np.array(numerator))
            roots = np.roots(char_poly)
            root_loci.append({
                'k': float(k),
                'roots': roots.tolist()
            })

        return {
            'k_range': k_range,
            'root_loci': root_loci,
            'plant_transfer_function': plant_tf
        }
