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
/- Attempt to import any generated artifact modules if present. This is a
   best-effort import that allows tests to add small example artifact files
   into `src/lean/LeanNiche/generated_artifacts/` and have lake compile them.
   If the imports file is missing, we silently ignore it so build remains robust.
/-
try
  import LeanNiche.generated_artifacts_imports
catch _ =>
  -- no generated artifacts
end


