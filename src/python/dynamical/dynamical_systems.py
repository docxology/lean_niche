#!/usr/bin/env python3
"""
Dynamical Systems Module for LeanNiche

This module provides comprehensive dynamical systems analysis including:
- Logistic map bifurcation analysis
- Chaos detection using Lyapunov exponents
- Nonlinear oscillator analysis
- Poincaré sections and phase portraits
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path


class DynamicalSystemsAnalyzer:
    """Base class for dynamical systems analysis."""

    def __init__(self):
        """Initialize the dynamical systems analyzer."""
        pass

    def lyapunov_exponent(self, trajectory: np.ndarray, dt: float = 1.0) -> float:
        """Compute Lyapunov exponent from trajectory data."""
        if len(trajectory) < 2:
            return 0.0

        # Simplified Lyapunov exponent calculation
        # In practice, this would involve computing the divergence of nearby trajectories
        divergence = 0.0
        for i in range(1, len(trajectory)):
            # Simple divergence estimate based on trajectory curvature
            if i < len(trajectory) - 1:
                v1 = trajectory[i] - trajectory[i-1]
                v2 = trajectory[i+1] - trajectory[i]
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                divergence += angle

        return divergence / (len(trajectory) * dt) if len(trajectory) > 0 else 0.0


class LogisticMapAnalyzer:
    """Analysis tools for the logistic map."""

    def __init__(self):
        """Initialize logistic map analyzer."""
        pass

    def analyze_logistic_map(self, r_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the logistic map for different parameter values.

        Args:
            r_params: Parameters for analysis (r_min, r_max, num_points, transient, iterations)

        Returns:
            Analysis results including bifurcation data and Lyapunov exponents
        """
        print("🧮 Analyzing logistic map dynamics...")

        # Extract parameters
        r_min = r_params.get('r_min', 2.5)
        r_max = r_params.get('r_max', 4.0)
        num_points = r_params.get('num_points', 300)
        transient = r_params.get('transient', 100)
        iterations = r_params.get('iterations', 200)

        r_values = np.linspace(r_min, r_max, num_points)

        results = {
            'bifurcation_data': [],
            'lyapunov_exponents': [],
            'chaos_regions': []
        }

        print("  📈 Computing bifurcation diagram...")
        for r in r_values:
            x = 0.5  # Initial condition

            # Iterate to remove transient
            for _ in range(transient):
                x = r * x * (1 - x)

            # Collect bifurcation points
            bifurcation_points = []
            for _ in range(iterations):
                x = r * x * (1 - x)
                if not np.isnan(x) and not np.isinf(x):
                    bifurcation_points.append(float(x))

            results['bifurcation_data'].append({
                'r': float(r),
                'points': bifurcation_points
            })

            # Compute Lyapunov exponent approximation
            if len(bifurcation_points) > 1:
                # Simple Lyapunov exponent calculation for logistic map
                lyapunov_sum = 0
                for i in range(1, min(50, len(bifurcation_points))):
                    if abs(bifurcation_points[i-1]) > 1e-10:
                        derivative_approx = abs(r * (1 - 2 * bifurcation_points[i-1]))
                        if derivative_approx > 0:
                            lyapunov_sum += np.log(derivative_approx)

                lyapunov_exp = lyapunov_sum / min(50, len(bifurcation_points))
                results['lyapunov_exponents'].append({
                    'r': float(r),
                    'lyapunov_exponent': float(lyapunov_exp)
                })

                # Detect chaos (positive Lyapunov exponent)
                if lyapunov_exp > 0.005:  # Small threshold
                    results['chaos_regions'].append(float(r))

        print(f"✅ Analyzed {len(r_values)} parameter values")
        return results

    def find_fixed_points(self, r: float) -> List[float]:
        """Find fixed points of the logistic map."""
        # Fixed points: x = r * x * (1 - x)
        # Solutions: x = 0 or x = 1 - 1/r
        fixed_points = [0.0]
        if r != 0:
            fp2 = 1 - 1/r
            if 0 <= fp2 <= 1:  # Physically meaningful
                fixed_points.append(fp2)
        return fixed_points

    def stability_analysis(self, r: float) -> Dict[str, Any]:
        """Analyze stability of fixed points."""
        fixed_points = self.find_fixed_points(r)
        stability_info = {}

        for i, fp in enumerate(fixed_points):
            # Derivative of logistic map: f'(x) = r * (1 - 2x)
            derivative = r * (1 - 2 * fp)

            if abs(derivative) < 1:
                stability = "stable"
            elif derivative < -1 or derivative > 1:
                stability = "unstable"
            else:
                stability = "marginally_stable"

            stability_info[f'fixed_point_{i}'] = {
                'value': fp,
                'derivative': derivative,
                'stability': stability
            }

        return stability_info


class NonlinearOscillatorAnalyzer:
    """Analysis tools for nonlinear oscillators."""

    def __init__(self):
        """Initialize nonlinear oscillator analyzer."""
        pass

    def analyze_nonlinear_oscillator(self, system_params: Dict[str, Any],
                                   initial_conditions: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze a nonlinear oscillator system.

        Args:
            system_params: System parameters (t_span, dt, alpha, beta, gamma)
            initial_conditions: List of (x0, y0) initial conditions

        Returns:
            Analysis results including trajectories and energy analysis
        """
        print("⚡ Analyzing nonlinear oscillator...")

        # Extract parameters
        t_span = system_params.get('t_span', (0, 50))
        dt = system_params.get('dt', 0.01)
        alpha = system_params.get('alpha', -1.0)  # x term coefficient
        beta = system_params.get('beta', -1.0)    # x^3 term coefficient
        gamma = system_params.get('gamma', -0.1)  # y term coefficient

        # Define nonlinear oscillator: dx/dt = y, dy/dt = alpha*x + beta*x^3 + gamma*y
        def nonlinear_oscillator(t: float, state: np.ndarray) -> np.ndarray:
            x, y = state
            dxdt = y
            dydt = alpha * x + beta * x**3 + gamma * y
            return np.array([dxdt, dydt])

        # Time parameters
        t = np.arange(t_span[0], t_span[1], dt)

        trajectories = []
        energies = []

        print("  📈 Computing trajectories...")
        for i, ic in enumerate(initial_conditions):
            print(f"    Trajectory {i+1}/{len(initial_conditions)}: x0={ic[0]}, y0={ic[1]}")

            # Simple Euler integration
            state = np.array(ic)
            trajectory = [state.copy()]
            energy = [self._compute_energy(state, alpha, beta)]  # Initial energy

            for _ in t[1:]:
                # Euler step
                derivative = nonlinear_oscillator(_, state)
                state = state + dt * derivative
                trajectory.append(state.copy())
                energy.append(self._compute_energy(state, alpha, beta))

            trajectories.append({
                'initial_condition': ic,
                'trajectory': np.array(trajectory),
                'energy': np.array(energy),
                'time': t
            })

        print("✅ Nonlinear oscillator analysis complete")
        return {
            'trajectories': trajectories,
            'system_parameters': system_params,
            'time': t,
            'system_name': f'Nonlinear Oscillator: dx/dt = y, dy/dt = {alpha}x + {beta}x³ + {gamma}y'
        }

    def _compute_energy(self, state: np.ndarray, alpha: float, beta: float) -> float:
        """Compute the energy of the nonlinear oscillator."""
        x, y = state
        # Energy for nonlinear oscillator: (1/2)y² + (alpha/2)x² + (beta/4)x^4
        kinetic = 0.5 * y**2
        potential = 0.5 * alpha * x**2 + 0.25 * beta * x**4
        return kinetic + potential

    def poincare_section(self, trajectory: np.ndarray, plane: str = 'x=0',
                        direction: str = 'positive') -> np.ndarray:
        """Compute Poincaré section of trajectory."""
        x_vals = trajectory[:, 0]
        y_vals = trajectory[:, 1]

        section_points = []

        if plane == 'x=0' and direction == 'positive':
            # Find crossings where x changes from negative to positive
            for i in range(1, len(x_vals)):
                if x_vals[i-1] < 0 and x_vals[i] >= 0:
                    section_points.append([x_vals[i], y_vals[i]])

        return np.array(section_points) if section_points else np.empty((0, 2))

    def frequency_analysis(self, trajectory: np.ndarray, time: np.ndarray) -> Dict[str, Any]:
        """Perform frequency analysis on trajectory."""
        # Simple FFT-based frequency analysis
        x_vals = trajectory[:, 0]
        dt = time[1] - time[0] if len(time) > 1 else 1.0

        # FFT
        fft_vals = np.fft.fft(x_vals)
        freqs = np.fft.fftfreq(len(x_vals), dt)

        # Find dominant frequency
        power_spectrum = np.abs(fft_vals)**2
        dominant_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC component
        dominant_freq = abs(freqs[dominant_idx])

        return {
            'frequencies': freqs.tolist(),
            'power_spectrum': power_spectrum.tolist(),
            'dominant_frequency': dominant_freq,
            'period': 1.0 / dominant_freq if dominant_freq > 0 else float('inf')
        }


class ChaosDetector:
    """Tools for chaos detection in dynamical systems."""

    def __init__(self):
        """Initialize chaos detector."""
        pass

    def detect_chaos(self, trajectory: np.ndarray, time: np.ndarray,
                    method: str = 'lyapunov') -> Dict[str, Any]:
        """Detect chaos in trajectory data."""
        if method == 'lyapunov':
            return self._lyapunov_chaos_detection(trajectory, time)
        elif method == 'correlation_dimension':
            return self._correlation_dimension_chaos_detection(trajectory)
        else:
            return {'chaos_detected': False, 'method': method, 'error': 'Unknown method'}

    def _lyapunov_chaos_detection(self, trajectory: np.ndarray, time: np.ndarray) -> Dict[str, Any]:
        """Chaos detection using Lyapunov exponent."""
        if len(trajectory) < 10:
            return {'chaos_detected': False, 'lyapunov_exponent': 0.0}

        # Simplified Lyapunov exponent calculation
        analyzer = DynamicalSystemsAnalyzer()
        lyap_exp = analyzer.lyapunov_exponent(trajectory, time[1] - time[0] if len(time) > 1 else 1.0)

        return {
            'chaos_detected': lyap_exp > 0.1,  # Threshold for chaos
            'lyapunov_exponent': lyap_exp,
            'method': 'lyapunov'
        }

    def _correlation_dimension_chaos_detection(self, trajectory: np.ndarray) -> Dict[str, Any]:
        """Chaos detection using correlation dimension."""
        # Simplified correlation dimension calculation
        if len(trajectory) < 50:
            return {'chaos_detected': False, 'correlation_dimension': 1.0}

        # Compute correlation dimension (simplified)
        distances = []
        for i in range(len(trajectory)):
            for j in range(i+1, len(trajectory)):
                dist = np.linalg.norm(trajectory[i] - trajectory[j])
                if dist > 0:
                    distances.append(dist)

        if not distances:
            return {'chaos_detected': False, 'correlation_dimension': 1.0}

        # Simple correlation dimension estimate
        r_values = np.logspace(-3, 0, 20)
        corr_dims = []

        for r in r_values:
            count = sum(1 for d in distances if d < r)
            if count > 0:
                corr_dims.append(np.log(count) / np.log(1/r))

        if corr_dims:
            avg_corr_dim = np.mean(corr_dims[-5:])  # Use last few values
            chaos_detected = avg_corr_dim > 2.0  # Fractal dimension > 2 suggests chaos
        else:
            avg_corr_dim = 1.0
            chaos_detected = False

        return {
            'chaos_detected': chaos_detected,
            'correlation_dimension': avg_corr_dim,
            'method': 'correlation_dimension'
        }
