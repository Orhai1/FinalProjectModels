import argparse, json, os, warnings
import numpy as np
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

warnings.filterwarnings("ignore")
RND = 0  # reproducibility

def make_scaler(X, ext_features, split_blocks=True):
    """StandardScaler on whole matrix OR per-block (video|aux)."""
    if not split_blocks or X.shape[1] <= ext_features:
        return StandardScaler()
    return ColumnTransformer([
        ("video", StandardScaler(), slice(0, ext_features)),
        ("aux",   StandardScaler(), slice(ext_features, None))
    ])

def run_search(name, model, param_dist, X, y, split_blocks, ext_features, n_iter=35, ):
    pipe = Pipeline([
        ("scale", make_scaler(X, ext_features, split_blocks)),
        ("clf",   model)
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
    search = RandomizedSearchCV(
        pipe,
        {f"clf__{k}": v for k, v in param_dist.items()},
        n_iter=n_iter,
        scoring=make_scorer(balanced_accuracy_score),
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RND
    )
    search.fit(X, y)
    print(f"[{name}]  best bal-acc = {search.best_score_:.4f}")
    print(json.dumps(search.best_params_, indent=2))
    return search.best_estimator_

def main(data_npz, out_dir, ext_features, split = True):
    d       = np.load(data_npz, allow_pickle=True)
    X, y_txt    = d["X"], d["y"]
    os.makedirs(out_dir, exist_ok=True)

    classes = sorted(np.unique(y_txt))
    class2idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class2idx[c] for c in y_txt], dtype=np.int16)

    xgb_space = dict(
        n_estimators      = randint(300, 900),
        learning_rate     = uniform(0.01, 0.19),
        max_depth         = randint(3, 9),
        subsample         = uniform(0.6, 0.4),
        colsample_bytree  = uniform(0.6, 0.4),
        gamma             = uniform(0, 5),
        min_child_weight  = randint(1, 8),
    )
    lgb_space = dict(
        n_estimators=randint(300, 900),
        learning_rate=uniform(0.01, 0.19),
        max_depth=randint(3, 9),
        num_leaves=randint(15, 63),
        subsample=uniform(0.6, 0.4),
        feature_fraction=uniform(0.6, 0.4),
        min_child_weight=uniform(1e-3, 0.3),
        min_split_gain=uniform(0, 0.5),
        min_child_samples=randint(5, 40),
        lambda_l2=uniform(0, 5),
        scale_pos_weight=uniform(1, 4)
    )

    xgb_best = run_search(
        "XGBoost",
        XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            tree_method='hist',
            n_jobs=-1,
            random_state=RND),
        xgb_space, X, y, split, ext_features)

    joblib.dump(xgb_best, os.path.join(out_dir, "xgb_tuned.joblib"))

    # lgb_best = run_search(
    #     "LightGBM",
    #     LGBMClassifier(
    #         objective='multiclass',
    #         random_state=RND,
    #         n_jobs=-1,
    #         is_unbalance=False,
    #         verbosity=-1),
    #     lgb_space, X, y, split, ext_features)
    #
    # joblib.dump(lgb_best, os.path.join(out_dir, "lgb_tuned.joblib"))
    print("Saved tuned models to", out_dir)

if __name__ == "__main__":
    data_npz = "data/bigru_early_fusion.npz"
    out_dir   = "models/bigru_fusion_tuned"
    num_ext_features = 768
    main(data_npz, out_dir, num_ext_features, split=True)
