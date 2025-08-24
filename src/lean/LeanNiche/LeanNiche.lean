/- LeanNiche root aggregation wrapper
   This file re-exports the core modules located under `src/lean/core/` so that
   imports like `LeanNiche.Basic` resolve when building the Lake package.
-/

import LeanNiche.Basic
import LeanNiche.Advanced
import LeanNiche.Statistics
import LeanNiche.LinearAlgebra
import LeanNiche.DynamicalSystems
import LeanNiche.Lyapunov
import LeanNiche.SetTheory
import LeanNiche.FreeEnergyPrinciple
import LeanNiche.ActiveInference
import LeanNiche.ControlTheory
import LeanNiche.SignalProcessing
import LeanNiche.Tactics
import LeanNiche.Utils
import LeanNiche.Visualization
import LeanNiche.Setup
import LeanNiche.PredictiveCoding
import LeanNiche.BeliefPropagation
import LeanNiche.DecisionMaking
import LeanNiche.LearningAdaptation
import LeanNiche.ProofExport

/- Minimal file to satisfy lake build -/


