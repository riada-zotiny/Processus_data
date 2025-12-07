import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Importation de ta fonction de chargement
from data.data_loader import load_data

# Configuration des graphiques
sns.set_palette("husl")

# Titre
st.title("☕ Exploration des données - Coffee Health")

# Chargement des données
df = load_data()
st.success("Données chargées avec succès")

# --- Aperçu des données ---
st.subheader("Aperçu des données")
st.dataframe(df.head())

# --- Infos générales ---
st.subheader("Informations générales")
st.write("Dimensions :", df.shape)
st.text(str(df.info()))

st.subheader("Valeurs manquantes")
st.dataframe(df.isnull().sum())

# --- Variables numériques ---
st.subheader("Analyse des variables numériques")

numeric_cols = [
    'Age', 'Coffee_Intake', 'Caffeine_mg', 'Sleep_Hours',
    'BMI', 'Heart_Rate', 'Physical_Activity_Hours', 'Smoking', 'Alcohol_Consumption'
]

stats = {}
for var in numeric_cols:
    stats[var] = {
        "Moyenne": df[var].mean(),
        "Médiane": df[var].median(),
        "Min": df[var].min(),
        "Max": df[var].max(),
        "Std": df[var].std()
    }

st.dataframe(pd.DataFrame(stats).T)

st.subheader("Histogrammes")
fig1 = df[numeric_cols].hist(figsize=(16, 12), bins=30)
st.pyplot(plt.gcf())

st.subheader("Boxplots")
plt.figure(figsize=(12,5))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=90)
st.pyplot(plt.gcf())

# --- Variables catégorielles ---
st.subheader("Analyse des variables catégorielles")

categorical_cols = [
    'Gender', 'Country', 'Sleep_Quality', 'Stress_Level',
    'Health_Issues', 'Occupation'
]

for col in categorical_cols:
    st.write(f"### {col}")
    st.dataframe(df[col].value_counts())

    plt.figure(figsize=(10,4))
    df[col].value_counts().head(10).plot(kind="bar")
    plt.title(col)
    st.pyplot(plt.gcf())

# --- Analyse bivariée ---
st.subheader("Analyse des relations entre variables")

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[17, 30, 45, 60, 80],
    labels=["18-30", "31-45", "46-60", "61-80"]
)

df["Gros_Buveur_Cafe"] = (df["Coffee_Intake"] > df["Coffee_Intake"].median()).map({True: "Oui", False: "Non"})

col1, col2 = st.columns(2)

with col1:
    st.write("Sommeil selon consommation de café")
    fig, ax = plt.subplots()
    sns.boxplot(x='Gros_Buveur_Cafe', y='Sleep_Hours', data=df, ax=ax)
    st.pyplot(fig)

with col2:
    st.write("BMI selon consommation de café")
    fig, ax = plt.subplots()
    sns.boxplot(x='Gros_Buveur_Cafe', y='BMI', data=df, ax=ax)
    st.pyplot(fig)

# --- Scatter plots ---
st.subheader("Relations graphiques")

fig, ax = plt.subplots()
ax.scatter(df['Caffeine_mg'], df['Sleep_Hours'], alpha=0.3)
ax.set_xlabel("Caféine (mg)")
ax.set_ylabel("Heures de sommeil")
st.pyplot(fig)

# --- Matrice de corrélation ---
st.subheader("Matrice de corrélation")

correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0)
st.pyplot(plt.gcf())

# --- Résumé ---
st.subheader("Résumé")

st.write(f"• Consommation moyenne de café : {df['Coffee_Intake'].mean():.2f} tasses/jour")
st.write(f"• Caféine moyenne : {df['Caffeine_mg'].mean():.1f} mg")
st.write(f"• Sommeil moyen : {df['Sleep_Hours'].mean():.2f} heures")
st.write(f"• BMI moyen : {df['BMI'].mean():.2f}")
st.write(f"• Âge moyen : {df['Age'].mean():.1f} ans")
st.write(f"• Niveau de stress le plus fréquent : {df['Stress_Level'].mode()[0]}")
