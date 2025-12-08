# Etapes pour pourvoir executer le projet

## Import du code

Il faut d'abord importer le code et les métadonnées dvc du git en utilisant la commande `git clone https://github.com/riada-zotiny/Processus_data`
Ceci copiera le projet dans le repertoire dans lequel vous vous trouvez lors de l'execution de la commande.

## Instalation des dépendances

Il faut ensuite créer un environnement python virtuel dans lequel on installera les différentes librairies utilisées pour cela dans la racine de votre projet executer les commandes suivantes :
1. `python -m venv .venv`
2. `./.venv/Scripts/activate` utilisez `\` à la place de `/` si vous êtes sur Windows
3. `pip install -r requirements.txt`

## Récupération des données

Pour cette partie il faudera se connecter au cloud avant de pouvoir récuperer les données.

### Connexion à AWS

Il faut installer AWS CLI via ce lien
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
Choisissez la version qui correspond a votre environnement puis sur une invite de commande saisissez la commande `aws configure`.
Cela vous demandera des identifiants que vous devrait demander en envoyant un mail à samimosta123@gmail.com (dans le cadre de l'évaluation du projet un compte a deja été crée et les identifiants vous ont été envoyé)

### Récupération des données

Enfin il suffit juste de récupérer les données en tappant la commande `dvc pull` à la racine du projet.


#  Guide d'utilisation d'un Notebook Jupyter

Ce notebook est conçu pour vous montrer comment interagir efficacement avec un autre notebook (ici, `01_exploration.ipynb`) et tirer parti de l'environnement Jupyter.

## 1. Objectif du Notebook

L'objectif de ce notebook est de présenter les étapes pour ouvrir, lire, exécuter, et modifier un fichier Jupyter Notebook existant.

## 2. Comment Ouvrir et Lire le Notebook `01_exploration.ipynb`

Un notebook est composé de cellules de deux types principaux : **Markdown** (texte formaté) et **Code**.

### Cellules Markdown

Ces cellules contiennent des explications, des titres, des descriptions, comme le README que vous avez vu précédemment.

- **Action** : Double-cliquez sur n'importe quelle cellule de texte (comme celle-ci) pour voir le code Markdown brut.  
- **But** : Comprendre les objectifs du projet, la description des données (GlobalCoffeeHealth dataset), les noms des auteurs, et les conclusions écrites.

### Cellules Code

Ces cellules contiennent du code (généralement Python) qui effectue les traitements, les calculs, et les visualisations.

## 3. Comment Exécuter le Code

L'exécution des cellules est essentielle pour reproduire les analyses.

### Exécution Séquentielle

1. Assurez-vous que l'environnement Python est prêt (le noyau/kernel est connecté).  
2. Exécutez la première cellule de code qui importe les bibliothèques (`pandas`, `matplotlib`, etc.) et charge les données (par exemple, un fichier `coffee_data.csv`).  

- **Action** : Cliquez sur la cellule de code et appuyez sur **Maj + Entrée** (ou utilisez le bouton "Exécuter" ▶️ dans la barre d'outils).  
- **Observation** : Un numéro (In [1], In [2], etc.) apparaîtra à gauche de la cellule pour indiquer qu'elle a été exécutée.

### Exécution du Notebook Entier

Pour exécuter tout le travail sans interruption :

- **Action** : Allez dans le menu **Noyau (ou Kernel)** et sélectionnez **Redémarrer et tout exécuter...** (ou *Restart & Run All*).  
- **But** : Cette méthode est la meilleure pour vérifier que toutes les étapes fonctionnent dans l'ordre et que les résultats finaux sont reproductibles.

## 4. Interprétation des Résultats

Dans `01_exploration.ipynb`, les cellules de code produisent :

- **Affichage de tables** : Utilisant `df.head()` pour voir les 5 premières lignes du jeu de données.  
- **Statistiques descriptives** : Utilisant `df.describe()` pour voir les moyennes, écarts-types, etc.  
- **Visualisations** : Les lignes de code créant des graphiques (`plt.hist()`, `sns.boxplot()`, etc.) affichent directement les figures dans la sortie de la cellule.

**Conseil** : Lisez toujours le texte Markdown juste avant une cellule de code pour comprendre l'objectif de ce code (par exemple, "Visualiser la distribution de l'IMC").









# Guide d'utilisation du Notebook `02_experiments.ipynb`

Ce notebook est un tutoriel pour comprendre et exécuter les expériences de Machine Learning contenues dans le fichier `02_experiments.ipynb`.

---

## 1. Objectifs du Notebook d'Expérimentation

Le notebook `02_experiments.ipynb` vise à :

- Entraîner et évaluer différents modèles de Machine Learning (ML) : **Random Forest (rf)**, **SVM**, et **MLP (Multi-Layer Perceptron)**.
- Utiliser la validation croisée (**GridSearchCV**) pour trouver les meilleurs hyperparamètres.
- Assurer la traçabilité et la reproductibilité des expériences grâce à **MLflow**.
- Sélectionner le meilleur modèle pour la prédiction de la qualité du sommeil.

---

## 2. Prérequis et Configuration

L'exécution de ce notebook dépend de deux éléments essentiels :

### A. Les Dépendances Python

Toutes les bibliothèques doivent être installées dans votre environnement virtuel. Le notebook nécessite l'importation des modules suivants :

- `sklearn` : pour `train_test_split`, `StandardScaler`, `MinMaxScaler`, les modèles et `GridSearchCV`.
- `mlflow` : pour la traçabilité des expériences.

### B. Le fichier `preprocessing.py`

Le notebook utilise un module externe `preprocessing.py` pour le prétraitement des données.  

**Action requise** : Assurez-vous que le fichier `preprocessing.py` est accessible (souvent dans le répertoire parent `..` ou le même répertoire que le notebook) et qu'il contient les fonctions nécessaires pour nettoyer et préparer les données avant l'entraînement.

---

## 3. Étapes d'Exécution et Compréhension du Code

Il est crucial d'exécuter les cellules dans l'ordre pour que les expériences ML se déroulent correctement.

### Étape 1 : Initialisation et Importations

- **Contenu** : Importation des bibliothèques (`sklearn`, `mlflow`) et configuration du chemin d'accès à `preprocessing.py`.
- **Action** : Exécutez cette cellule (`Maj + Entrée`).
- **Résultat attendu** : Le message `"Les bibliothèques sont importées avec succès."` doit s'afficher.

### Étape 2 : Prétraitement et Séparation des Données

- **Contenu** : Appel des fonctions du module `preprocessing.py` pour charger, nettoyer, normaliser les données et les séparer en ensembles d'entraînement/test (`X_train`, `X_test`, `y_train`, `y_test`).

### Étape 3 : Lancement des Expériences MLflow

- **Contenu** : Les cellules suivantes contiennent des boucles ou blocs de code pour l'entraînement des modèles (Random Forest, SVM, MLP) en utilisant `GridSearchCV`. Chaque exécution est enregistrée par MLflow.
- **Action** : Exécutez chaque cellule. Le temps d'exécution peut varier selon la taille du dataset et les hyperparamètres testés.

### Étape 4 : Visualisation des Scores

- **Contenu** : Une cellule affichant le dictionnaire `best_scores`.
- **Résultat attendu** : Exemple d'affichage des scores finaux :

```python
{'rf': 0.80436..., 'svm': 0.80436..., 'mlp': 0.80772...}
```

## 4. Interprétation et Suivi avec MLflow

Le cœur de ce notebook est la **traçabilité des expériences**.

### Rôle de MLflow

MLflow permet de stocker pour chaque modèle entraîné :

- **Paramètres** : Hyperparamètres testés (ex. profondeur de l'arbre pour Random Forest).  
- **Métriques** : Performances (Score F1 pondéré, Précision, Rappel).  
- **Artefacts** : Le modèle lui-même et d'autres fichiers (ex. graphiques).

### Conclusion du Notebook

Comme indiqué dans la dernière cellule de `02_experiments.ipynb`, l'analyse a conduit à la sélection du **SVM avec un noyau linéaire** comme modèle optimal, obtenant un **score de 99.26%**.






# Guide d'utilisation du Notebook `03_explainability.ipynb`

Ce notebook est conçu pour vous aider à comprendre comment interpréter les décisions du modèle optimal sélectionné dans `02_experiments.ipynb`.

---

## 1. Objectifs Réels du Notebook

Malgré un objectif d'exploration réutilisé par erreur dans le Markdown, le véritable objectif de ce notebook est **l'explicabilité (Explainability)** :

- **Interpréter le Modèle Optimal** : Utiliser des techniques avancées pour comprendre les mécanismes de décision du modèle SVM entraîné (ou tout autre modèle final).  
- **Identifier l'Importance des Caractéristiques** : Déterminer quelles variables (IMC, heures de sommeil, caféine, etc.) ont le plus d’impact sur la prédiction de la qualité du sommeil.  
- **Fournir des Explications Locales** : Comprendre pourquoi le modèle a fait une prédiction spécifique pour un individu donné.

---

## 2. Outils et Prérequis

Ce notebook dépend de l'étape précédente et nécessite des outils spécifiques pour l'interprétabilité.

### A. Le Modèle Entraîné

Le notebook doit charger le modèle optimal (**SVM avec noyau linéaire**) qui a été sauvegardé lors de l'exécution de `02_experiments.ipynb` (via MLflow ou un fichier `.pkl`).

### B. La Librairie SHAP

Le code utilise la librairie **SHAP** (SHapley Additive exPlanations), méthode courante pour attribuer la contribution de chaque variable à la prédiction.

**Action requise** : Assurez-vous que la librairie `shap` est installée dans votre environnement Python.

### C. Les Données Prétraitées

Comme dans le notebook précédent, les données doivent être chargées et prétraitées (via `preprocessing.py`) pour alimenter le modèle dans le même format utilisé pour l'entraînement.

---

## 3. Étapes d'Exécution et Compréhension du Code

L'exécution doit être **séquentielle**, car chaque étape dépend de la précédente.

### Étape 1 : Importations et Chargement du Modèle

- **Contenu** : Importation des bibliothèques (`pandas`, `numpy`, `matplotlib`, `shap`) et initialisation des chemins d'accès. Chargement du modèle sauvegardé.  
- **Action** : Exécutez cette cellule (`Maj + Entrée`).  
- **Résultat attendu** : Le message `"Les bibliothèques sont importées avec succès."` doit s'afficher.

### Étape 2 : Calcul des Valeurs SHAP

- **Contenu** : Création de l'objet **SHAP Explainer** adapté au modèle chargé et calcul des valeurs de Shapley pour un sous-ensemble des données de test.  
- **But** : Ce calcul, intensif, permet de déterminer l’impact de chaque variable sur la prédiction de la qualité du sommeil.

### Étape 3 : Visualisation Globale (Feature Importance)

- **Contenu** : Utilisation de la fonction `shap.summary_plot()`.  
- **Résultat attendu** : Un graphique résumant l’impact global de chaque variable.  
- **Lecture du graphique** :  
  - Les variables en haut sont les plus importantes.  
  - La couleur (du bleu au rouge) montre si une valeur élevée de cette variable a un impact **positif ou négatif** sur la prédiction (ex. impact positif sur une bonne qualité de sommeil).


#  Guide d'utilisation du Notebook `streamlit_app.ipynb`
## Guide rapide : utilisation de l'application Streamlit

### Lancement
- Activer l'environnement et lancer l'app :
  ```
  .venv\Scripts\activate
  streamlit run streamlit_app.py
  ```

### Pages principales

- Single Prediction : widgets auto-générés depuis un exemple aléatoire du dataset. L'entrée est prétraitée avec `src/data/preprocessing.py` avant prédiction.


- Batch Prediction : upload d'un CSV ou utilisation d'un échantillon. Après chargement, la colonne `Sleep_Quality` est supprimée pour l'affichage et les colonnes manquantes (ID, Country, Age, Health_Issues, Sleep_Quality) sont ajoutées si nécessaire avant le prétraitement.

### Sortie & téléchargement
- Après batch, le tableau de sortie contient : `prediction` (encodée), `prediction_label` (lisible) et colonnes `prob_*` si `predict_proba` est disponible. Bouton pour télécharger CSV.

### Remarques
- Le mapping des labels lisibles utilise : `{'Poor':0,'Fair':1,'Good':2,'Excellent':3}`.
- Pour la production, envisager de sauvegarder le préprocesseur (joblib) afin d'éviter de refiter les encodeurs à chaque requête.






