
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://tasks:tasks@localhost:5432/tasks",
)

EXPERIMENT_NAME = "task-priority-classifier"
REGISTERED_MODEL_NAME = "task-priority-classifier"


def load_dataset() -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)

    df = pd.read_sql(
        "SELECT title, description, priority FROM tasks WHERE priority IS NOT NULL",
        engine,
    )

    if df.empty:
        raise RuntimeError(
            "No labeled tasks found in the database. Seed some with "
            "`python model/generate_data.py`, or create tasks with a `priority` "
            "set through the API, before training."
        )

    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

    return df


def main() -> None:
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["priority"],
        test_size=0.2,
        random_state=42,
        stratify=df["priority"],
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(max_features=500, ngram_range=(1, 2)),
        ),
        (
            "clf",
            LogisticRegression(max_iter=1000),
        ),
    ])

    with mlflow.start_run():

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_param("model_type", "tfidf+logreg")
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1)

        # Create DataFrame input example
        input_example = pd.DataFrame({
            "text": X_train.iloc[:2].tolist()
        })

        # Generate predictions and model signature
        output_example = pipeline.predict(input_example["text"])

        signature = infer_signature(
            input_example,
            output_example,
        )

        # Log and register model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=input_example,
            signature=signature,
        )

        print(f"accuracy={accuracy:.3f} f1_macro={f1:.3f}")


if __name__ == "__main__":
    main()

