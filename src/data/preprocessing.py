import pandas as pd
from sklearn.preprocessing import LabelEncoder

def add_age_groups(df):
    bins = [0, 18, 30, 45, 60, 120]
    labels = ["0-18", "19-30", "31-45", "46-60", "60+"]
    df = df.copy()
    df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)
    return df

def preprocess_data(df):
    # Aperçu des données
    print(df.head())
    print(f"\nDimensions: {df.shape}")
    print(f"\nTypes de données:\n{df.dtypes}")
    print(f"\nValeurs manquantes:\n{df.isnull().sum()}")

    # Stats des variables numériques
    num_vars = ['Age', 'Coffee_Intake', 'Caffeine_mg', 'Sleep_Hours', 
                'BMI', 'Heart_Rate', 'Physical_Activity_Hours', 'Smoking', 'Alcohol_Consumption']

    for var in num_vars:
        print(f"\n{var}:")
        print(f"  Moyenne: {df[var].mean():.2f}")
        print(f"  Médiane: {df[var].median():.2f}")
        print(f"  Écart-type: {df[var].std():.2f}")
        print(f"  Min: {df[var].min():.2f} | Max: {df[var].max():.2f}")

    # Répartition des variables catégorielles
    cat_vars = ['Gender', 'Country', 'Sleep_Quality', 'Stress_Level', 
                'Health_Issues', 'Occupation']

    for var in cat_vars:
        print(f"\n{var}:")
        counts = df[var].value_counts()
        percentages = (counts / len(df)) * 100
        for cat, count in counts.items():
            print(f"  {cat}: {count} ({percentages[cat]:.1f}%)")

    # Résumé des indicateurs clés
    print(f"\n   • Consommation moyenne de café: {df['Coffee_Intake'].mean():.2f} tasses/jour")
    print(f"   • Caféine moyenne: {df['Caffeine_mg'].mean():.1f} mg")
    print(f"   • Sommeil moyen: {df['Sleep_Hours'].mean():.2f} heures")
    print(f"   • BMI moyen: {df['BMI'].mean():.2f}")
    print(f"   • Âge moyen: {df['Age'].mean():.1f} ans")
    print(f"   • Niveau de stress le plus fréquent: {df['Stress_Level'].mode()[0]}")
    
    # Traitement des valeurs manquantes
    print(df['Health_Issues'].value_counts())
    df["Health_Issues"] = df["Health_Issues"].fillna("None")

    # Création de features géographiques et démographiques
    df = add_age_groups(df)
    
    continent_map = {
        'France': 'Europe', 'Germany': 'Europe', 'Spain': 'Europe', 'Italy': 'Europe',
        'USA': 'America', 'Canada': 'America', 'Brazil': 'America', 'Mexico': 'America',
        'Japan': 'Asia', 'China': 'Asia', 'India': 'Asia', 'South Korea': 'Asia',
        'Australia': 'Oceania', 'New Zealand': 'Oceania',
        'South Africa': 'Africa', 'Morocco': 'Africa', 'Egypt': 'Africa'
    }
    df['Continent'] = df['Country'].map(continent_map).fillna('Other')

    # Suppression des colonnes redondantes
    df.drop(columns=['ID', 'Country', 'Age'], inplace=True)

    # Encodage ordinal des variables avec hiérarchie naturelle
    sleep_quality_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
    stress_level_map = {'Low': 0, 'Medium': 1, 'High': 2}
    health_issues_map = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
    age_groups_map = {'0-18': 0, '19-30': 1, '31-45': 2, '46-60': 3, '60+': 4}

    df['Age_Group'] = df['Age_Group'].map(age_groups_map)
    df['Sleep_Quality'] = df['Sleep_Quality'].map(sleep_quality_map)
    df['Stress_Level'] = df['Stress_Level'].map(stress_level_map)
    df['Health_Issues'] = df['Health_Issues'].map(health_issues_map)

    # Label encoding pour les variables catégorielles nominales
    label_encoders = {}
    cat_vars = ['Gender', 'Continent', 'Occupation']

    for col in cat_vars:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Affichage des mappings
    print("\nMapping Sleep_Quality:", sleep_quality_map)
    print("Mapping Stress_Level:", stress_level_map)
    print("Mapping Health_Issues:", health_issues_map)

    for col, le in label_encoders.items():
        print(f"\nMapping pour {col}:")
        for original, encoded in zip(le.classes_, le.transform(le.classes_)):
            print(f"  {original}: {encoded}")
    
    return df