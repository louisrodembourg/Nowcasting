import pandas as pd
import matplotlib.pyplot as plt

# 📂 Charger ton fichier
df = pd.read_parquet("data/parquet/market/cl_f_2019-01-01_2019-12-31.parquet")

# 🧹 Sécurité
df = df.sort_values("date")

# 📈 Performance cumulée (base 100)
df["cum_return"] = (1 + df["return"].fillna(0)).cumprod()
df["cum_return"] = df["cum_return"] * 100

# 🎨 Plot
plt.figure()
plt.plot(df["date"], df["cum_return"])
plt.xlabel("Date")
plt.ylabel("Performance (base 100)")
plt.title("XLE Performance Over Time")
plt.grid()

plt.show()