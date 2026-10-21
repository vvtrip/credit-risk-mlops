from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import make_preprocessor


def build_models(frame, random_state: int) -> dict[str, Pipeline]:
    preprocessor = make_preprocessor(frame)
    logistic = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
    ])
    forest = Pipeline([
        ("preprocess", make_preprocessor(frame)),
        ("classifier", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )),
    ])
    boosting = Pipeline([
        ("preprocess", make_preprocessor(frame)),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=random_state,
        )),
    ])
    stack = Pipeline([
        ("preprocess", make_preprocessor(frame)),
        ("classifier", StackingClassifier(
            estimators=[
                ("logistic", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ("forest", RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced",
                                                   random_state=random_state, n_jobs=-1)),
                ("boosting", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                                              max_leaf_nodes=15, random_state=random_state)),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            stack_method="predict_proba",
            cv=5,
            n_jobs=-1,
        )),
    ])
    return {"logistic_baseline": logistic, "random_forest_bagging": forest,
            "hist_gradient_boosting": boosting, "stacking_ensemble": stack}


def calibrate(model, cv: int = 3):
    return CalibratedClassifierCV(model, method="sigmoid", cv=cv)
