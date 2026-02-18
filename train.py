import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import mlflow

# 1. Load Data
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["species"] = iris.target
data.to_csv("iris.csv", index=False)

X = data.drop("species", axis=1)
y = data["species"]

# 2. Split Data
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# 3. MLflow Tracking Block
with mlflow.start_run():
    # --- LOG PARAMETERS (Settings you chose) ---
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("max_iter", 300)

    # --- TRAIN MODEL ---
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    # --- LOG METRICS (The results) ---
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # --- LOG ARTIFACTS (The actual files) ---
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")
    mlflow.log_artifact("iris.csv")

print(f"Model trained with accuracy: {accuracy}")