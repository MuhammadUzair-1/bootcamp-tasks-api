import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

model_name = "task-priority-classifier"
model_uri =  "models:/task-priority-classifier/3"

model = mlflow.sklearn.load_model(model_uri)

test_cases = [
    ("Production server is down", "The application is unavailable for all users.", "HIGH"),
    ("Update documentation", "Improve the project README and add examples.", "LOW"),
    ("Database performance is slow", "Investigate and improve query execution time.", "MEDIUM"),
    ("Critical security vulnerability", "A serious vulnerability needs immediate attention.", "HIGH"),
    ("Fix minor UI alignment", "Correct a small spacing issue on the dashboard.", "LOW"),
]

for title, description, expected in test_cases:
    text = pd.Series([title + " " + description])
    prediction = model.predict(text)[0]

    print(f"Task:      {title}")
    print(f"Expected:  {expected}")
    print(f"Predicted: {prediction}")
    print("-" * 45)
