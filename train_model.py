import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

print("Loading data for training...")
df = pd.read_csv("novagen_dataset.csv")

# Ensure bool columns are ints
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

X = df.drop("Target", axis=1)
y = df["Target"]

print("Training final XGBoost model...")
xgb_clf = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)
xgb_clf.fit(X, y)

print("Deploying model to xgboost_health_model.pkl...")
joblib.dump(xgb_clf, "xgboost_health_model.pkl")

# Save expected columns so we know exactly the order for predictions
joblib.dump(list(X.columns), "model_columns.pkl")
print("Training Complete. Web model is ready!")
