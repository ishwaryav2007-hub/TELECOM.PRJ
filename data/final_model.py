import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/5G_energy_consumption_dataset.csv"
)

print("Rows:", len(df))


# ==========================================
# 2. CONVERT TIME
# ==========================================

df["Time"] = pd.to_datetime(
    df["Time"],
    format="%Y%m%d %H%M%S"
)

df["hour"] = df["Time"].dt.hour
df["dayofweek"] = df["Time"].dt.dayofweek


# ==========================================
# 3. SORT CHRONOLOGICALLY
# ==========================================

df = df.sort_values("Time")


# ==========================================
# 4. FEATURES
# ==========================================

features = [
    "load",
    "ESMODE",
    "TXpower",
    "hour",
    "dayofweek"
]

target = "Energy"


# ==========================================
# 5. TIME-BASED SPLIT
# ==========================================

split_index = int(len(df) * 0.80)

train = df.iloc[:split_index]
test = df.iloc[split_index:]


X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


print("\nTraining rows:", len(train))
print("Testing rows:", len(test))

print("\nTraining period:")
print(train["Time"].min(), "to", train["Time"].max())

print("\nTesting period:")
print(test["Time"].min(), "to", test["Time"].max())


# ==========================================
# 6. TRAIN MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

print("\nTraining final model...")

model.fit(X_train, y_train)

print("Training complete!")


# ==========================================
# 7. PREDICTION
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 8. EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n==============================")
print("TIME-BASED MODEL PERFORMANCE")
print("==============================")

print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))


# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(importance)


# ==========================================
# 10. SAVE FINAL MODEL
# ==========================================

joblib.dump(
    model,
    "energy_model_final.pkl"
)

print("\nFinal model saved as:")
print("energy_model_final.pkl")