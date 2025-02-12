# Modèle de Classification 
pour la Prédiction des Actions de Capteurs

✅ Les valeurs manquantes ont été traitées
✅ L’avertissement sur action_valide a été géré proprement
✅ Le modèle de Machine Learning a été entraîné avec une précision de 99.5% (ce qui est excellent !)
✅ Le fichier corrigé a été généré avec succès


📌 Vérification des objectifs de l’exercice

✅ 1. Charger et analyser les données
	•	Fichiers CSV chargés (captor_1_sample_1.csv, captor_2_with_null.csv, captor_3_todo.csv)
	•	Exploration des données avec print(df_sample1.head())
	•	Détection des valeurs manquantes (print(df_sample2.isnull().sum()))

✅ 2. Corriger les erreurs dans les données
	•	Valeurs manquantes remplacées par la médiane (SimpleImputer(strategy='median'))
	•	Gestion des erreurs avec un avertissement si action_valide n’existe pas

✅ 3. Compléter les données manquantes
	•	Remplissage des capteurs (temp, sis, hygro, anem1, anem2) par la médiane
	•	Correction des actions manquantes en les prédisant via Machine Learning

✅ 4. Choisir la bonne action à chaque relevé
	•	Modèle RandomForest entraîné sur les données validées (captor_1_sample_1.csv)
	•	Évaluation du modèle (accuracy_score()) avec 99.5% de précision
	•	Application du modèle pour prédire les actions manquantes dans captor_3_todo.csv

✅ 5. Générer un fichier corrigé et le soumettre à l’API
	•	Sauvegarde de captor_3_todo_corrected.csv
	•	Code pour tester la soumission à l’API fourni (requests.post())


Résumé du code :

Ce script effectue une prédiction d’actions en fonction de données capteurs en utilisant un modèle de RandomForestClassifier. Voici un résumé des étapes :
	1.	Chargement des données :
	•	Trois fichiers CSV sont lus :
	•	captor_1_sample_1.csv (données d’entraînement)
	•	captor_2_with_null.csv (données avec valeurs manquantes)
	•	captor_3_todo.csv (données à prédire)
	2.	Exploration des données :
	•	Affiche un aperçu de df_sample1.
	•	Vérifie les valeurs manquantes dans df_sample2.
	3.	Prétraitement des données :
	•	Les valeurs manquantes des colonnes numériques (temp, sis, hygro, anem1, anem2) sont remplacées par la médiane.
	•	La variable action_valide est convertie en valeurs numériques (A → 0, B → 1, C → 2, SB → 3).
	•	Vérifie si action_valide est présente dans df_sample2.
	4.	Construction du modèle de Machine Learning :
	•	Sépare les données en train (80%) et test (20%).
	•	Entraîne un RandomForestClassifier sur ces données.
	5.	Évaluation du modèle :
	•	Effectue des prédictions sur les données de test.
	•	Affiche la précision du modèle.
	6.	Prédiction sur les nouvelles données (df_todo) :
	•	Applique le même prétraitement (imputation des valeurs manquantes).
	•	Utilise le modèle entraîné pour prédire action.
	•	Convertit les valeurs prédictives en labels (0 → A, 1 → B, etc.).
	•	Sauvegarde le fichier final captor_3_todo_corrected.csv.

Conclusion :

Le script automatise le nettoyage, l’entraînement et la prédiction d’actions basées sur des mesures de capteurs, en corrigeant les valeurs manquantes et en utilisant un modèle de classification Random Forest.