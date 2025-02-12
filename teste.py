import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les fichiers CSV
file_sample1 = "captor_1_sample_1.csv"
file_sample2 = "captor_2_with_null.csv"
file_todo = "captor_3_todo.csv"

df_sample1 = pd.read_csv(file_sample1)
df_sample2 = pd.read_csv(file_sample2)
df_todo = pd.read_csv(file_todo)

# Exploration des données
print("Aperçu des données de sample_1:")
print(df_sample1.head())

# Vérifier les valeurs manquantes
print("Valeurs manquantes dans sample_2:")
print(df_sample2.isnull().sum())

# Remplacement des valeurs manquantes par la médiane des colonnes numériques
imputer = SimpleImputer(strategy='median')
num_cols = ['temp', 'sis', 'hygro', 'anem1', 'anem2']
df_sample2[num_cols] = imputer.fit_transform(df_sample2[num_cols])

# Encoder les actions en valeurs numériques
action_mapping = {'A': 0, 'B': 1, 'C': 2, 'SB': 3}
df_sample1['action_valide'] = df_sample1['action_valide'].map(action_mapping)

# Vérification si 'action_valide' existe dans df_sample2
if 'action_valide' in df_sample2.columns:
    df_sample2['action_valide'] = df_sample2['action_valide'].map(action_mapping)
else:
    print("⚠️ Attention : 'action_valide' n'existe pas dans sample_2. On ignore cette étape.")

# Définir les variables et les cibles
X = df_sample1[num_cols]
y = df_sample1['action_valide']

# Séparer les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
predictions = model.predict(X_test)
print("Précision du modèle:", accuracy_score(y_test, predictions))

# Appliquer le modèle aux nouvelles données
df_todo[num_cols] = imputer.transform(df_todo[num_cols])
df_todo['action'] = model.predict(df_todo[num_cols])

# Convertir les prédictions en valeurs d'actions
reverse_mapping = {v: k for k, v in action_mapping.items()}
df_todo['action'] = df_todo['action'].map(reverse_mapping)

# Sauvegarder le fichier final
df_todo.to_csv("captor_3_todo_corrected.csv", index=False)
print("Fichier captor_3_todo_corrected.csv généré avec succès!")
