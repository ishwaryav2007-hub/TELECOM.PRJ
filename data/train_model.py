import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/5G_energy_consumption_dataset.csv")

print("Dataset loaded!")
print("Rows:", len(df))


# ==========================================
# 2. CONVERT TIME
# ==========================================

df["Time"] = pd.to_datetime(
    df["Time"],
    format="%Y%m%d %H%M%S"
)

# Extract useful time information
df["hour"] = df["Time"].dt.hour
df["day"] = df["Time"].dt.day
df["dayofweek"] = df["Time"].dt.dayofweek


# ==========================================
# 3. SELECT FEATURES
# ==========================================

features = [
    "load",
    "ESMODE",
    "TXpower",
    "hour",
    "dayofweek"
]

target = "Energy"

X = df[features]
y = df[target]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. CREATE MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 6. TRAIN
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete!")


# ==========================================
# 7. PREDICT
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 8. EVALUATE
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(y_test, predictions)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))


# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(importance)


# ==========================================
# 10. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "energy_model.pkl"
)

print("\nModel saved as energy_model.pkl")