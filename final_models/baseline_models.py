import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedShuffleSplit, train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix
)
from imblearn.ensemble import BalancedRandomForestClassifier

from final_models.LabelEncoderWrapper import LabelEncoderWrapper


def _stratified_sample(X, y, n_total, seed=42):
    # If n_total is larger than the number of samples, return all indices
    if n_total >= len(y):
        return np.arange(len(y))
    # Otherwise, perform stratified sampling (preserves class proportions and doesn't repeat samples)
    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_total,
        random_state=seed
    )
    idx_small, _ = next(sss.split(X, y))
    return idx_small


def load_features(npz_path,
                  sample_size=None,
                  random_seed=42):
    """Load features .npz and optionally down-sample."""
    data = np.load(npz_path)
    X, y, ids = data["X"], data["y"], data["video_id"]
    if sample_size:
        idx = _stratified_sample(X, y, sample_size, random_seed)
        return X[idx], y[idx], ids[idx]
    return X, y, ids


def make_logreg_pipeline():
    """Create a pipeline with StandardScaler and LogisticRegression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='lbfgs',
                           max_iter=10_000,
                           class_weight="balanced")
    )


def make_balanced_rf(random_seed=42):
    """Create a BalancedRandomForestClassifier."""
    return BalancedRandomForestClassifier(
        n_estimators=400,
        sampling_strategy="auto",
        n_jobs=1,
        random_state=random_seed
    )


def make_logreg_with_smote(random_seed=42):
    """
    Pipeline: [SMOTE] ➜ [StandardScaler] ➜ [LogisticRegression]
    SMOTE is applied *only* on the training folds inside CV.
    """
    smote = SMOTE(random_state=random_seed, k_neighbors=5)
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=10_000, class_weight=None)
    return Pipeline(steps=[
        ("smote", smote),
        ("scale", scaler),
        ("clf"  , clf)
    ])

def make_rf_with_smote(random_seed=42):
    """
    Pipeline: [SMOTE] ➜ [RandomForest]
    Uses default RF hyper-params except n_estimators & class_weight=None.
    """
    smote = SMOTE(random_state=random_seed, k_neighbors=5)

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,          # tune later if you like
        n_jobs=1,
        random_state=random_seed,
        class_weight=None        # let SMOTE do the balancing
    )

    return Pipeline([
        ("smote", smote),
        ("rf", rf),
    ])

def make_xgb(random_seed=42, use_smote=False):
    xgb_core = xgb.XGBClassifier(
        objective      = "multi:softprob",
        learning_rate  = 0.08,
        n_estimators   = 400,
        max_depth      = 6,
        subsample      = 0.9,
        colsample_bytree = 0.8,
        eval_metric    = "mlogloss",
        tree_method    = "hist",
        n_jobs         = -1,
        random_state   = random_seed
    )

    wrapped = LabelEncoderWrapper(xgb_core)

    if use_smote:
        return Pipeline([
            ("smote", SMOTE(random_state=random_seed)),
            ("xgb"  , wrapped)
        ])
    else:
        return wrapped

def train_test_once(X, y, model, test_size=0.2, seed=42):
    """Train-test split and evaluate the model."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    metrics = {
        "balanced_acc": balanced_accuracy_score(y_te, y_pred),
        "report": classification_report(y_te, y_pred, digits=3),
        "confusion": confusion_matrix(y_te, y_pred)
    }
    return metrics


def cv_score(X, y, model, k=5, seed=42):
    """Cross-validation score for a given model."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y,
                             scoring="balanced_accuracy",
                             cv=skf,
                             n_jobs=1)
    return scores.mean(), scores.std()
