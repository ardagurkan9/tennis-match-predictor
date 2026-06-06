"""Model training utilities."""

from pathlib import Path

import joblib
import pandas as pd
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
]


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


def train_logistic_regression() -> tuple[Pipeline, dict[str, dict[str, float]], Path]:
    """Train and evaluate the first logistic regression model."""
    train, validation, test = load_split_data()

    x_train, y_train = split_features_and_target(train)
    x_validation, y_validation = split_features_and_target(validation)
    x_test, y_test = split_features_and_target(test)

    model = build_logistic_regression_model()
    model.fit(x_train, y_train)

    metrics = {
        "validation": evaluate_classifier(model, x_validation, y_validation),
        "test": evaluate_classifier(model, x_test, y_test),
    }
    model_path = save_model(model)

    return model, metrics, model_path


def main() -> None:
    """Train the logistic regression model."""
    _, metrics, model_path = train_logistic_regression()

    print("Logistic regression model")
    print("-------------------------")
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


if __name__ == "__main__":
    main()
