import pandas as pd

df = pd.read_csv("data/5G_energy_consumption_dataset.csv")

print("\n========== ESMODE ==========")

print(df.groupby("ESMODE")["Energy"].agg(
    ["count", "mean", "min", "max"]
))


print("\n========== TXPOWER ==========")

print(df.groupby("TXpower")["Energy"].agg(
    ["count", "mean", "min", "max"]
))


print("\n========== LOAD vs ENERGY ==========")

print(
    df[["load", "Energy"]].corr()
)