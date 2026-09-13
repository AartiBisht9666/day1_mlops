
import os
import pandas as pd
import mlflow
import joblib

from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# 2. MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

experiment_name = "Advertising_Sales_Regression"
registered_model_name = "Sales_Prediction_Model"

mlflow.set_experiment(experiment_name)


# ============================================================
# 3. DATA PREPARATION
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Data file missing at {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

X = df[["TV", "radio", "newspaper"]]
y = df["sales"]

xtrain, xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 4. TRAIN CANDIDATE MODELS
# ============================================================

models = {
    "Linear_Regression": LinearRegression(),
    "Ridge_Regression": Ridge(alpha=1.0),
    "Random_Forest": RandomForestRegressor(
        max_depth=5,
        random_state=42
    )
}

batch_runs = []


for name, model in models.items():

    with mlflow.start_run(run_name=name) as run:

        # Train model
        model.fit(xtrain, ytrain)

        # Predictions
        predictions = model.predict(xtest)

        # RMSE
        rmse = root_mean_squared_error(
            ytest,
            predictions
        )

        # Log parameters and metrics
        mlflow.log_param(
            "model_type",
            name
        )

        mlflow.log_metric(
            "test_rmse",
            rmse
        )

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        # Store run ID, RMSE and trained model
        batch_runs.append(
            (
                run.info.run_id,
                rmse,
                model
            )
        )

        print(
            f"{name} trained successfully | "
            f"RMSE: {rmse:.4f}"
        )


# ============================================================
# 5. FIND BEST MODEL
# ============================================================

# Lower RMSE = better model
batch_runs.sort(
    key=lambda x: x[1]
)

best_run_id, best_rmse, best_model = batch_runs[0]

print()
print(
    f"🏆 Best model selected | "
    f"Run ID: {best_run_id} | "
    f"RMSE: {best_rmse:.4f}"
)


# ============================================================
# 6. REGISTER CHALLENGER MODEL
# ============================================================

client = MlflowClient()

challenger_model = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=registered_model_name
)

challenger_version = challenger_model.version

client.set_registered_model_alias(
    registered_model_name,
    "challenger",
    challenger_version
)

print(
    f"🥊 Challenger registered: "
    f"Version {challenger_version} "
    f"(RMSE: {best_rmse:.4f})"
)


# ============================================================
# 7. CHALLENGER VS CHAMPION EVALUATION
# ============================================================

try:

    champion_info = client.get_model_version_by_alias(
        registered_model_name,
        "champion"
    )

    champion_run = client.get_run(
        champion_info.run_id
    )

    champion_rmse = champion_run.data.metrics[
        "test_rmse"
    ]

    champion_version = champion_info.version

    print(
        f"Current Champion: "
        f"Version {champion_version} "
        f"(RMSE: {champion_rmse:.4f})"
    )

    # Lower RMSE wins
    if best_rmse < champion_rmse:

        client.set_registered_model_alias(
            registered_model_name,
            "champion",
            challenger_version
        )

        print(
            f"🏆 Title Change! "
            f"Challenger (v{challenger_version}) "
            f"defeated Champion (v{champion_version})"
        )

    else:

        print(
            f"🛡️ Champion (v{champion_version}) "
            f"defended title."
        )

except Exception:

    # First model becomes champion
    client.set_registered_model_alias(
        registered_model_name,
        "champion",
        challenger_version
    )

    print(
        f"🌟 First Champion assigned: "
        f"Version {challenger_version}"
    )


# ============================================================
# 8. EXPORT CHAMPION MODEL
# ============================================================

# IMPORTANT:
# We directly save the best trained model instead of
# loading it again from the MLflow Registry.
#
# This avoids the CT error:
# "No such artifact: 'MLmodel'"

champion_export_path = os.path.join(
    MODELS_DIR,
    "champion_model.pkl"
)

joblib.dump(
    best_model,
    champion_export_path
)

print()
print(
    f"✅ Champion model exported to "
    f"{champion_export_path}"
)

