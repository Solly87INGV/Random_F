import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)

###################################
# Impostazioni percorso per salvataggio output PNG e CSV
###################################
output_dir = r"E:\INGV\1_Human Mobility\2_FindCircular_Algorithm\R&RandomForerst\Results_RF+R_script"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

###################################
# 1) Lettura del file originale OmanData.xlsx
###################################
file_path = r"E:\INGV\1_Human Mobility\2_FindCircular_Algorithm\R&RandomForerst\OmanData.xlsx"
df_oman = pd.read_excel(file_path, sheet_name=0)
print("OmanData columns:", df_oman.columns)

###################################
# 2) Lettura dei risultati dello script R
###################################
results_folder = r"E:\INGV\1_Human Mobility\2_FindCircular_Algorithm\R&RandomForerst\Results_R_script"

df_kmeans   = pd.read_csv(os.path.join(results_folder, "kmeans_results.csv"), sep=";")
df_dbscan   = pd.read_csv(os.path.join(results_folder, "dbscan_results.csv"), sep=";")
df_hotspot  = pd.read_csv(os.path.join(results_folder, "hotspot_results.csv"), sep=";")
df_localmor = pd.read_csv(os.path.join(results_folder, "local_moran_df_results.csv"), sep=";")

# Pulizia dei nomi delle colonne: rimuove spazi e virgolette
def clean_columns(df):
    df.columns = df.columns.str.strip().str.replace('"', '')
    return df

df_kmeans   = clean_columns(df_kmeans)
df_dbscan   = clean_columns(df_dbscan)
df_hotspot  = clean_columns(df_hotspot)
df_localmor = clean_columns(df_localmor)

print("kmeans_results columns:", df_kmeans.columns)
print("dbscan_results columns:", df_dbscan.columns)
print("hotspot_results columns:", df_hotspot.columns)
print("local_moran_df_results columns:", df_localmor.columns)

###################################
# 3) Merge su "ID"
###################################
df_merged = pd.merge(df_oman, df_kmeans, on='ID', how='left')
df_merged = pd.merge(df_merged, df_dbscan, on='ID', how='left')
df_merged = pd.merge(df_merged, df_hotspot, on='ID', how='left')
df_merged = pd.merge(df_merged, df_localmor, on='ID', how='left')

# Se il merge ha creato colonne duplicate (es. "cluster" in entrambi), rinominale:
df_merged.rename(columns={'cluster_x': 'kmeans_cluster', 'cluster_y': 'dbscan_cluster'}, inplace=True)

print("Colonne dopo il merge:", df_merged.columns)
print("Numero di record finali:", df_merged.shape[0])

###################################
# 4) Analisi Esplorativa (opzionale)
###################################
print("\nPrimi 5 record del dataset unito:")
print(df_merged.head(), "\n")
print("Informazioni sul dataset unito:")
print(df_merged.info(), "\n")
print("Statistiche descrittive del dataset unito:")
print(df_merged.describe(), "\n")

###################################
# 5) Pre-elaborazione e Preparazione dei Dati
###################################
# Target: 'Class'
# Feature originali (dal dataset OmanData)
original_features = ['Diameter', 'Width', 'Lenght', 'Height']

# Nuove feature derivate dallo script R:
# - kmeans_results.csv fornisce "kmeans_cluster"
# - dbscan_results.csv fornisce "dbscan_cluster"
# - hotspot_results.csv fornisce "Gi_star"
# - local_moran_df_results.csv fornisce "Ii" e "Z.Ii"
new_features = ['kmeans_cluster', 'dbscan_cluster', 'Gi_star', 'Ii', 'Z.Ii']

# Rimuove i record senza 'Class'
df_merged.dropna(subset=['Class'], inplace=True)

# Trasforma gli 0 in NaN nelle feature originali (se applicabile)
df_merged[original_features] = df_merged[original_features].replace(0, np.nan)

# Converte le nuove feature da stringa a numerico sostituendo la virgola con il punto
for col in ['Gi_star', 'Ii', 'Z.Ii']:
    df_merged[col] = pd.to_numeric(df_merged[col].str.replace(",", "."), errors="coerce")

# Creazione di X e y
all_features = original_features + new_features
X = df_merged[all_features].values
y = df_merged['Class'].values

# Label Encoding della variabile target
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

###################################
# 6) Suddivisione Train/Test
###################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

###################################
# 7) Pipeline (imputazione + RandomForest)
###################################
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

pipeline = Pipeline([
    ('impute', imputer),
    ('rf', model_rf)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

###################################
# 8) Risultati e salvataggio grafico
###################################
acc = accuracy_score(y_test, y_pred)
print("\n=== RANDOM FOREST con Feature Originali + Feature da R ===")
print("Accuracy:", acc)
print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

# Bar Plot per precision, recall e F1-score per classe
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=range(len(encoder.classes_))
)
classes = encoder.classes_
x_vals = np.arange(len(classes))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x_vals - width, precision, width, label='Precision')
rects2 = ax.bar(x_vals, recall, width, label='Recall')
rects3 = ax.bar(x_vals + width, f1, width, label='F1-score')

ax.set_ylabel('Score')
ax.set_title('Metriche di Classificazione per Classe')
ax.set_xticks(x_vals)
ax.set_xticklabels(classes)
ax.legend()

for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
plt.savefig(os.path.join(output_dir, "classification_metrics.png"))
plt.show()

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred, labels=range(len(encoder.classes_)))
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title("Matrice di Confusione (Feature Originali + da R)")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.show()

###################################
# 9) Salvataggio del dataset unito in CSV ben strutturato
###################################
merged_csv_path = os.path.join(output_dir, "merged_dataset.csv")
# Salva il dataset unito usando ";" come delimitatore
df_merged.to_csv(merged_csv_path, index=False, sep=";")
print(f"Dataset unito salvato in: {merged_csv_path}")
