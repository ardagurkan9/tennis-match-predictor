"""Model training utilities."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "target_win"

NON_FEATURE_COLUMNS = [
    TARGET_COLUMN,
    "match_id",
    "tourney_id",
    "tourney_name",
    "tourney_date",
    "match_num",
    "player_id",
    "opponent_id",
    "player_name",
    "opponent_name",
    "player_ioc",
    "opponent_ioc",
    "ioc_matchup",
    "draw_size",
]

DATASET_DIRECTORIES = {
    "base": Path("data/features"),
    "advanced": Path("data/features/advanced"),
}

MODEL_DIRECTORIES = {
    "base": Path("models"),
    "advanced": Path("models/advanced"),
}


def load_split_data(
    train_path: str | Path = "data/features/train.csv",
    validation_path: str | Path = "data/features/validation.csv",
    test_path: str | Path = "data/features/test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test feature splits."""
    return (
        pd.read_csv(train_path),
        pd.read_csv(validation_path),
        pd.read_csv(test_path),
    )


def load_split_data_from_directory(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all feature splits from a dataset directory."""
    data_dir = Path(data_dir)
    return load_split_data(
        train_path=data_dir / "train.csv",
        validation_path=data_dir / "validation.csv",
        test_path=data_dir / "test.csv",
    )


def split_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split model input features and target."""
    columns_to_drop = [
        column for column in NON_FEATURE_COLUMNS
        if column in data.columns
    ]

    x = data.drop(columns=columns_to_drop)
    y = data[TARGET_COLUMN]

    return x, y


def build_logistic_regression_model() -> Pipeline:
    """Build the first logistic regression model pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_model() -> Pipeline:
    """Build the random forest model pipeline."""
    return Pipeline(
        steps=[
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def build_xgboost_model() -> Pipeline:
    """Build the XGBoost model pipeline."""
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError(
            "xgboost is required to train the XGBoost model. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from error

    return Pipeline(
        steps=[
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_lightgbm_model() -> Pipeline:
    """Build the LightGBM model pipeline."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise ImportError(
            "lightgbm is required to train the LightGBM model. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from error

    return Pipeline(
        steps=[
            (
                "model",
                LGBMClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary",
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )


def evaluate_classifier(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Evaluate a binary classifier with standard metrics."""
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)[:, 1]

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
    }


def save_model(
    model: Pipeline,
    output_path: str | Path = "models/logistic_regression.pkl",
) -> Path:
    """Save a trained model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def train_logistic_regression(
    data_dir: str | Path = "data/features",
    model_dir: str | Path = "models",
) -> tuple[Pipeline, dict[str, dict[str, float]], Path]:
    """Train and evaluate the first logistic regression model."""
    train, validation, test = load_split_data_from_directory(data_dir)

    x_train, y_train = split_features_and_target(train)
    x_validation, y_validation = split_features_and_target(validation)
    x_test, y_test = split_features_and_target(test)

    model = build_logistic_regression_model()
    model.fit(x_train, y_train)

    metrics = {
        "validation": evaluate_classifier(model, x_validation, y_validation),
        "test": evaluate_classifier(model, x_test, y_test),
    }
    model_path = save_model(model, output_path=Path(model_dir) / "logistic_regression.pkl")

    return model, metrics, model_path


def train_random_forest(
    data_dir: str | Path = "data/features",
    model_dir: str | Path = "models",
) -> tuple[Pipeline, dict[str, dict[str, float]], Path]:
    """Train and evaluate the random forest model."""
    train, validation, test = load_split_data_from_directory(data_dir)

    x_train, y_train = split_features_and_target(train)
    x_validation, y_validation = split_features_and_target(validation)
    x_test, y_test = split_features_and_target(test)

    model = build_random_forest_model()
    model.fit(x_train, y_train)

    metrics = {
        "validation": evaluate_classifier(model, x_validation, y_validation),
        "test": evaluate_classifier(model, x_test, y_test),
    }
    model_path = save_model(model, output_path=Path(model_dir) / "random_forest.pkl")

    return model, metrics, model_path


def train_xgboost(
    data_dir: str | Path = "data/features",
    model_dir: str | Path = "models",
) -> tuple[Pipeline, dict[str, dict[str, float]], Path]:
    """Train and evaluate the XGBoost model."""
    train, validation, test = load_split_data_from_directory(data_dir)

    x_train, y_train = split_features_and_target(train)
    x_validation, y_validation = split_features_and_target(validation)
    x_test, y_test = split_features_and_target(test)

    model = build_xgboost_model()
    model.fit(x_train, y_train)

    metrics = {
        "validation": evaluate_classifier(model, x_validation, y_validation),
        "test": evaluate_classifier(model, x_test, y_test),
    }
    model_path = save_model(model, output_path=Path(model_dir) / "xgboost.pkl")

    return model, metrics, model_path


def train_lightgbm(
    data_dir: str | Path = "data/features",
    model_dir: str | Path = "models",
) -> tuple[Pipeline, dict[str, dict[str, float]], Path]:
    """Train and evaluate the LightGBM model."""
    train, validation, test = load_split_data_from_directory(data_dir)

    x_train, y_train = split_features_and_target(train)
    x_validation, y_validation = split_features_and_target(validation)
    x_test, y_test = split_features_and_target(test)

    model = build_lightgbm_model()
    model.fit(x_train, y_train)

    metrics = {
        "validation": evaluate_classifier(model, x_validation, y_validation),
        "test": evaluate_classifier(model, x_test, y_test),
    }
    model_path = save_model(model, output_path=Path(model_dir) / "lightgbm.pkl")

    return model, metrics, model_path


def print_model_report(
    model_name: str,
    metrics: dict[str, dict[str, float]],
    model_path: Path,
) -> None:
    """Print model training and evaluation results."""
    print(model_name)
    print("-" * len(model_name))
    print(f"Model saved to: {model_path}")

    for split_name, split_metrics in metrics.items():
        print(f"\n{split_name.title()} metrics")
        for metric_name, metric_value in split_metrics.items():
            if metric_name == "confusion_matrix":
                print(f"{metric_name}:")
                print(f"  {metric_value[0]}")
                print(f"  {metric_value[1]}")
            else:
                print(f"{metric_name}: {metric_value:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse the feature dataset selected for model training."""
    parser = argparse.ArgumentParser(description="Train all model candidates.")
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_DIRECTORIES),
        default="base",
        help="Feature dataset to train on (default: base).",
    )
    return parser.parse_args()


def main() -> None:
    """Train all current model candidates."""
    args = parse_args()
    data_dir = DATASET_DIRECTORIES[args.dataset]
    model_dir = MODEL_DIRECTORIES[args.dataset]

    print(f"Dataset: {args.dataset} ({data_dir})")
    print(f"Model output directory: {model_dir}")
    print()

    _, logistic_metrics, logistic_model_path = train_logistic_regression(data_dir, model_dir)
    _, random_forest_metrics, random_forest_model_path = train_random_forest(data_dir, model_dir)
    _, xgboost_metrics, xgboost_model_path = train_xgboost(data_dir, model_dir)
    _, lightgbm_metrics, lightgbm_model_path = train_lightgbm(data_dir, model_dir)

    print_model_report(
        "Logistic regression model",
        logistic_metrics,
        logistic_model_path,
    )
    print()
    print_model_report(
        "Random forest model",
        random_forest_metrics,
        random_forest_model_path,
    )
    print()
    print_model_report(
        "XGBoost model",
        xgboost_metrics,
        xgboost_model_path,
    )
    print()
    print_model_report(
        "LightGBM model",
        lightgbm_metrics,
        lightgbm_model_path,
    )


if __name__ == "__main__":
    main()
