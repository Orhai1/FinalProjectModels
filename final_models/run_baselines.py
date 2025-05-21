import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from baseline_models import (
    load_features,
    make_logreg_pipeline,
    make_balanced_rf,
    train_test_once,
    cv_score, make_logreg_with_smote, make_rf_with_smote, make_xgb,
)

def run_all_baselines(npz_path,
                      scale=False,
                      sample_size=None,
                      test_size=0.20,
                      kfold=5,
                      seed=42):
    """
    Train & evaluate several baseline models, then print a comparison table.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file produced by vectorize_vid.
    sample_size : int or None
        If set, first draw a stratified subset of this many samples.
    test_size : float
        Fraction of the (sub-)dataset reserved for the hold-out test.
    kfold : int
        Number of folds for Stratified K-Fold CV.
    seed : int
        Random seed for every stochastic step.
    """

    # Load the feature set and optionally down-sample it
    X, y, ids = load_features(npz_path, sample_size, seed)

    # Define baselines to compare
    models = {
        "LogisticReg": make_logreg_pipeline(scale), # linear + class_weight
        "BalancedRF":  make_balanced_rf(seed, scale), # non-linear, imbalance-aware
        "LogReg+SMOTE": make_logreg_with_smote(seed, scale), # linear + SMOTE
        "RF+SMOTE":  make_rf_with_smote(seed), # non-linear + SMOTE
        "XGBoost":  make_xgb(seed), # non-linear + class_weight,
        "XGBoost+SMOTE": make_xgb(seed, use_smote=True), # non-linear + SMOTE
    }

    results = []   # collect a row per model for the summary table

    # Loop over all models and evaluate them
    for name, model in models.items():
        print(f"\n――――  {name}  ――――")

        # perform one train/test split and evaluate the model
        split_metrics = train_test_once(
            X, y, model, test_size=test_size, seed=seed
        )
        print("Hold-out balanced-accuracy:", split_metrics["balanced_acc"])
        print(split_metrics["report"])

        # perform k-fold CV for a more robust estimate
        mean_cv, std_cv = cv_score(X, y, model, k=kfold, seed=seed)
        print(f"{kfold}-fold CV balanced-acc : {mean_cv:.3f} ± {std_cv:.3f}")

        # stash for the comparison table
        results.append({
            "model": name,
            "split_bal_acc": split_metrics["balanced_acc"],
            "cv_bal_acc_mean": mean_cv,
            "cv_bal_acc_std": std_cv
        })

    # sort the results by mean CV accuracy
    summary = (
        pd.DataFrame(results)
          .sort_values("cv_bal_acc_mean", ascending=False)
          .reset_index(drop=True)
    )
    print("\n=====  Summary (sorted by CV mean)  =====")
    print(summary.to_string(index=False, float_format="%.3f"))

    return summary


if __name__ == "__main__":
    run_all_baselines(npz_path="data/bigru_early_fusion.npz",
                      scale=True,
                      sample_size=None,
                      test_size=0.2,
                      kfold=5,
                      seed=42)
