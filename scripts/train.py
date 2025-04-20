import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# Load data
df = pd.read_csv("data/housing.csv")
X = df.drop("medv", axis=1)
y = df["medv"]
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Train and track
with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
    print(f"Model logged in run with MSE: {mse:.3f}")
