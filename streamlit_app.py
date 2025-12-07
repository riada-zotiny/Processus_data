# ...existing code...
import mlflow
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import glob, os, joblib, pickle

DATA_SAMPLE_PATH = "data/raw/synthetic_coffee_health_10000.csv"
TARGET_COLUMN = "Sleep_Quality"

def load_model_auto():
    tried = []
    candidates = glob.glob("**/model.pkl", recursive=True)
    candidates += glob.glob("../mlruns/**/*.pkl", recursive=True)
    mlflow_dirs = glob.glob("../mlruns/*/*/artifacts/*") + glob.glob("../mlruns/*/models/*")
    tried.extend(candidates + mlflow_dirs)

    for path in candidates:
        try:
            model = joblib.load(path)
            st.write(f"Model loaded (joblib) from: {path}")
            return model
        except Exception:
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                st.write(f"Model loaded (pickle) from: {path}")
                return model
            except Exception:
                continue

    for d in mlflow_dirs:
        try:
            model = mlflow.sklearn.load_model(d)
            st.write(f"Model loaded (mlflow) from: {d}")
            return model
        except Exception:
            pkl_inside = glob.glob(os.path.join(d, "*.pkl"))
            for p in pkl_inside:
                try:
                    model = joblib.load(p)
                    st.write(f"Model loaded (joblib) from inside {d}: {p}")
                    return model
                except Exception:
                    continue
            continue

    raise FileNotFoundError(f"No model found. Tried: {tried}")

def load_sample_dataset():
    if os.path.exists(DATA_SAMPLE_PATH):
        return pd.read_csv(DATA_SAMPLE_PATH)
    # fallback: try parent path
    alt = "../" + DATA_SAMPLE_PATH
    if os.path.exists(alt):
        return pd.read_csv(alt)
    return None

def build_input_widgets_from_df(df, exclude=[TARGET_COLUMN]):
    features = [c for c in df.columns if c not in exclude]
    values = {}
    for c in features:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            default = float(col.mean()) if not np.isnan(col.mean()) else 0.0
            values[c] = st.number_input(c, value=default)
        else:
            uniques = col.dropna().unique().tolist()
            if len(uniques) > 50:
                # too many categories -> free text
                values[c] = st.text_input(c, value=str(uniques[0]) if uniques else "")
            else:
                values[c] = st.selectbox(c, options=uniques)
    return features, values

def single_prediction_page(model, df_sample):
    st.header("Single Prediction")
    st.write("Formulaire généré automatiquement à partir du dataset analysé.")
    features, values = build_input_widgets_from_df(df_sample)
    if st.button("Predict"):
        # build row in same order as `features`
        row = []
        for f in features:
            v = values[f]
            # try convert categorical strings to numeric when possible
            try:
                row.append(float(v))
            except Exception:
                row.append(v)
        X = np.array(row).reshape(1, -1)
        try:
            pred = model.predict(X)
            st.success(f"Prediction: {pred[0]}")
            if hasattr(model, "predict_proba"):
                st.write("Probabilities:")
                st.write(model.predict_proba(X))
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

def batch_prediction_page(model, df_sample):
    st.header("Batch Prediction")
    st.info("Vous pouvez uploader un CSV pré-traité compatible avec le modèle ou utiliser un échantillon du dataset analysé.")
    use_sample = st.checkbox("Utiliser échantillon dataset analysé", value=False)
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
    df = None
    if use_sample:
        df = df_sample.copy()
        st.write("Aperçu de l'échantillon :")
        st.dataframe(df.head())
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu du fichier uploadé :")
        st.dataframe(df.head())
    else:
        st.write("Aucun fichier sélectionné.")

    if df is not None and st.button("Run Predictions"):
        try:
            preds = model.predict(df.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in df.columns else df)
            df["Prediction"] = preds
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df.drop(columns=["Prediction"]) if "Prediction" in df.columns else df)
                for i in range(proba.shape[1]):
                    df[f"Prob_Class_{i}"] = proba[:, i]
            st.success("Prédictions terminées")
            st.dataframe(df.head(20))
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger résultats CSV", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Erreur durant la prédiction batch : {e}")

def main():
    st.title("Machine Learning Prediction App — Dataset monitored")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Dataset"])

    df_sample = load_sample_dataset()
    if df_sample is None:
        st.warning(f"Dataset sample introuvable à {DATA_SAMPLE_PATH}. Assurez-vous que le fichier existe.")
    else:
        st.sidebar.write(f"Dataset sample loaded: {DATA_SAMPLE_PATH}")
        st.sidebar.write(f"Dimensions: {df_sample.shape}")

    # Load model
    try:
        model = load_model_auto()
    except Exception as e:
        st.error(f"Model file not found. {e}")
        st.write("Exemples de fichiers trouvés sous ../mlruns :")
        st.write(glob.glob("../mlruns/**/*", recursive=True)[:30])
        return

    if page == "Dataset":
        if df_sample is not None:
            st.header("Dataset analysé (aperçu)")
            st.dataframe(df_sample.head(50))
            st.write("Colonnes :")
            st.write(df_sample.columns.tolist())
        else:
            st.info("Aucun dataset sample disponible.")
    elif page == "Single Prediction":
        if df_sample is None:
            st.warning("Pour générer automatiquement le formulaire, placez le dataset analysé dans data/raw/... ou upload manuellement en Batch.")
        single_prediction_page(model, df_sample if df_sample is not None else pd.DataFrame())
    else:
        batch_prediction_page(model, df_sample if df_sample is not None else pd.DataFrame())

if __name__ == "__main__":
    main()
# ...existing code...