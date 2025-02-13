import pandas as pd
import numpy as np
import requests

# Pour le machine learning
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer

# ================================================================
#                 Définition des chemins de fichiers
# ================================================================
file_training = "captor_1_sample_2_with_manual_validation.csv"  # Contient action_valide (corrigé manuellement)
file_with_null = "captor_2_with_null.csv"                       # Exemple pour l'imputation
file_todo = "captor_3_todo.csv"                                 # Données à corriger + prédire

# ================================================================
#                      Chargement des données
# ================================================================
df_training = pd.read_csv(file_training)
df_null = pd.read_csv(file_with_null)
df_todo = pd.read_csv(file_todo)

# ================================================================
#         Vérification de la présence de valeurs manquantes
# ================================================================
def check_missing(df, df_name):
    print(f"Valeurs manquantes dans {df_name}:")
    print(df.isnull().sum(), "\n")

check_missing(df_training, "df_training")
check_missing(df_null, "df_null")
check_missing(df_todo, "df_todo")

# ================================================================
#        Imputation des valeurs manquantes via KNNImputer
# ================================================================
num_cols = ["temp", "sis", "hygro", "anem1", "anem2"]

imputer = KNNImputer(n_neighbors=5)

# On corrige les valeurs manquantes dans df_training et df_null
df_training[num_cols] = imputer.fit_transform(df_training[num_cols])
df_null[num_cols] = imputer.transform(df_null[num_cols])
df_todo[num_cols] = imputer.transform(df_todo[num_cols])

# ================================================================
#      Mapping des actions (A=0, B=1, C=2, SB=3) pour l'entraînement
# ================================================================
action_mapping = {"A": 0, "B": 1, "C": 2, "SB": 3}
reverse_mapping = {v: k for k, v in action_mapping.items()}

# On s'assure que la colonne 'action_valide' existe pour l'entraînement
if "action_valide" not in df_training.columns:
    raise ValueError("La colonne 'action_valide' n'existe pas dans df_training. Impossible de s'entraîner.")

df_training["action_valide"] = df_training["action_valide"].map(action_mapping)

# ================================================================
#                 Construction du dataset d'entraînement
# ================================================================
X = df_training[num_cols]
y = df_training["action_valide"]

# Séparation en jeu d'entraînement / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================================================
#           Entraînement du modèle (XGBoost de préférence)
# ================================================================
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# ================================================================
#         Évaluation du modèle sur le jeu de test interne
# ================================================================
preds_test = model.predict(X_test)
score = accuracy_score(y_test, preds_test)
print("Précision du modèle (Accuracy) sur le test set:", score)

# ================================================================
#        Application du modèle sur le fichier TODO
# ================================================================
df_todo["action"] = model.predict(df_todo[num_cols])
df_todo["action"] = df_todo["action"].map(reverse_mapping)

# ================================================================
#      Sauvegarde du fichier corrigé avec la colonne 'action'
# ================================================================
output_file = "captor_3_todo_corrected.csv"
df_todo.to_csv(output_file, index=False)
print(f"Fichier '{output_file}' généré avec succès !")

# ================================================================
#             Soumission du fichier corrigé à l'API
# ================================================================
url = "http://20.216.208.68"  # URL de l'API fournie
try:
    with open(output_file, "rb") as f:
        response = requests.post(url, files={"file": f})
    print("Résultat de l'API:", response.json())
except Exception as e:
    print("Erreur lors de l'appel à l'API :", e)