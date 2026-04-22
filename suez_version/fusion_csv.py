import pandas as pd

# Charger les fichiers
df1 = pd.read_csv("C:\\Users\\hadri\\OneDrive\\Documents\\Documents\\Enseignement\\UTC\\Cours\\P26_TZ\\Nowcasting_suez\\data\\suez_03_21.csv")
df2 = pd.read_csv("C:\\Users\\hadri\\OneDrive\\Documents\\Documents\\Enseignement\\UTC\\Cours\\P26_TZ\\Nowcasting_suez\\data\\suez_04_21.csv")
df3 = pd.read_csv("C:\\Users\\hadri\\OneDrive\\Documents\\Documents\\Enseignement\\UTC\\Cours\\P26_TZ\\Nowcasting_suez\\data\\suez_05_21.csv")

# Fusion (concaténation verticale)
df_final = pd.concat([df1, df2, df3], ignore_index=True)

# Sauvegarder le résultat
df_final.to_csv("C:\\Users\\hadri\\OneDrive\\Documents\\Documents\\Enseignement\\UTC\\Cours\\P26_TZ\\Nowcasting_suez\\data\\fusion.csv", index=False)