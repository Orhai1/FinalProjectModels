from scipy.stats import uniform, randint
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

BAL_ACC = make_scorer(balanced_accuracy_score)

def wrap_search(pipe, param_grid, random=True, n_iter=30, cv=5, seed=42):
    """
    Create a search object for hyperparameter tuning.
    """
    if random:
        return RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=BAL_ACC,
            cv=cv,
            n_jobs=-1,
            random_state=seed,
            verbose=1
        )
    else:
        return GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=BAL_ACC,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

SEARCH_SPACES = {
    "LogisticReg": {
        "clf__C":        uniform(0.1, 9.9),      # 0.1 – 10
        "clf__solver":   ["lbfgs", "saga"],
        "clf__penalty":  ["l2"],
    },
    "BalancedRF": {
        "clf__n_estimators":    randint(200, 800),
        "clf__max_depth":       [None, 8, 12],
        "clf__max_features":    ["sqrt", 0.7],
        "clf__min_samples_leaf":[1, 3, 5],
    },
    "XGBoost": {
        "clf__model__n_estimators":    randint(300, 900),
        "clf__model__learning_rate":   uniform(0.03, 0.17),
        "clf__model__max_depth":       randint(4, 9),
        "clf__model__subsample":       uniform(0.7, 0.3),
        "clf__model__colsample_bytree":uniform(0.7, 0.3),
    }
}
