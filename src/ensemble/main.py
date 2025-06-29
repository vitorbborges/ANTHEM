import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ensemble.optimization import OptimizationRunner

if __name__ == "__main__":
    print("🚀 Starting STABLE nested optimization...")
    print("Key improvements:")
    print("- Simplified Universal Kriging with fallback to drift-only models")
    print("- Better numerical stability checks")
    print("- Reduced complexity to prevent overfitting")
    print("- Enhanced error handling in nested optimization")
    print()

    # Your setup with reduced complexity for stability
    runner = OptimizationRunner(
        data_path="data/processed_data/combined_subjects.parquet",
        temporal_folds=4,  # 4 CV folds (16 subjects split into 4 groups of 4)
        spatial_folds=3,  # Reduced from 10 for stability
        holdout_size=0.2,  # 4 out of 20 subjects (20%) for final testing
        n_trials_per_subject=30,  # Reduced from 15 for faster, more stable optimization
    )

    print("Configuration:")
    print("- 20 subjects total")
    print("- 4 subjects in holdout set")
    print("- 16 subjects for cross-validation")
    print("- Each CV fold: 12 training subjects + 4 validation subjects")
    print("- 10 optimization trials per subject (reduced for stability)")
    print("- 3 spatial folds per subject (reduced for stability)")
    print("- Enhanced numerical stability checks")
    print()

    # Run optimization with fewer trials initially to test stability
    study = runner.run(n_trials=20)  # Start with 5 trials to test

    print("\n🎉 Optimization completed!")
    print("If results look good, you can increase n_trials and n_trials_per_subject")
    print("\nCheck the following files:")
    print("- models/best_ensemble_model.pkl")
    print("- metrics/best_model_analysis.json")
    print("- models/ensemble_trial_*.pkl for individual trial models")
