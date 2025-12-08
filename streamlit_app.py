# ...existing code...
import mlflow
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import glob, os, joblib, pickle
from src.data.preprocessing  import preprocess_data

DATA_SAMPLE_PATH = "data/raw/synthetic_coffee_health_10000.csv"
TARGET_COLUMN = "Sleep_Quality"
MODEL_PATH = "../mlruns/619235882260501448/models/m-2f93ce56213f4209ad48ba5c91a98c07/artifacts"

def load_model_auto():
    # 1) try the exact path specified by the user
    tried = []
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        st.write(f"Model loaded (mlflow) from explicit path: {MODEL_PATH}")
        return model
    except Exception as e:
        tried.append(MODEL_PATH)

    # 2) fallback to previous discovery logic
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


# ...existing code...
def build_input_widgets_from_df(df, exclude=[TARGET_COLUMN]):
    """
    Build widgets and pre-fill them with values taken from a random row of `df`
    (if available). If df is empty or a column is missing, fall back to previous defaults.
    Returns (features, values) where `values` contains the widget-returned values.
    """
    features = [c for c in df.columns if c not in exclude] if isinstance(df, pd.DataFrame) else []
    values = {}

    # try to pick a random row from the dataset to use as default values
    random_row = None
    if isinstance(df, pd.DataFrame) and not df.empty:
        try:
            random_row = df.sample(n=1).iloc[0]
        except Exception:
            random_row = None

    for c in features:
        # safe column access (handle case where df may be empty DataFrame passed)
        col = df[c] if (isinstance(df, pd.DataFrame) and c in df.columns) else pd.Series(dtype="float")
        if pd.api.types.is_numeric_dtype(col):
            # prefer the random row value if available and numeric
            if random_row is not None and pd.notna(random_row.get(c)):
                try:
                    default = float(random_row[c])
                except Exception:
                    default = float(col.mean()) if (not col.empty and not np.isnan(col.mean())) else 0.0
            else:
                default = float(col.mean()) if (not col.empty and not np.isnan(col.mean())) else 0.0
            values[c] = st.number_input(c, value=default)
        else:
            uniques = col.dropna().unique().tolist()
            if len(uniques) > 50:
                # too many categories -> free text, prefill from random row if present
                if random_row is not None and pd.notna(random_row.get(c)):
                    default_text = str(random_row[c])
                else:
                    default_text = str(uniques[0]) if uniques else ""
                values[c] = st.text_input(c, value=default_text)
            else:
                options = uniques if uniques else [""]
                # choose index so the selectbox shows the random row value if present
                if random_row is not None and pd.notna(random_row.get(c)) and random_row[c] in options:
                    default_index = options.index(random_row[c])
                else:
                    default_index = 0
                values[c] = st.selectbox(c, options=options, index=default_index)
    return features, values


# mapping for Sleep_Quality (same as in preprocessing)
sleep_quality_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
# inverse map for displaying labels from numeric predictions
sleep_quality_inv = {v: k for k, v in sleep_quality_map.items()}

# helper to get readable class label from model class identifier
def readable_label(cls):
    # if class is numeric (encoded), map to readable label
    try:
        if isinstance(cls, (int, np.integer, float, np.floating)):
            return sleep_quality_inv.get(int(cls), str(cls))
    except Exception:
        pass
    # if class is string and matches an original label, return it
    if isinstance(cls, str) and cls in sleep_quality_map:
        return cls
    return str(cls)

def single_prediction_page(model, df_sample):
    st.header("Single Prediction")
    st.write("Formulaire généré automatiquement à partir du dataset analysé.")
    features, values = build_input_widgets_from_df(df_sample)
    if st.button("Predict"):
        # build a single-row dict from widget values
        row_dict = {f: values[f] for f in features}

        try:
            # Combine row with sample dataset so preprocess_data fits encoders consistently.
            if isinstance(df_sample, pd.DataFrame) and not df_sample.empty:
                combined = pd.concat([df_sample.reset_index(drop=True), pd.DataFrame([row_dict])], ignore_index=True)
            else:
                combined = pd.DataFrame([row_dict])

            # preprocess the combined dataframe (this will map/encode columns as in src/data/preprocessing.py)
            processed = preprocess_data(combined)

            # Drop Sleep_Hours if present (as requested)
            if "Sleep_Hours" in processed.columns:
                processed = processed.drop(columns=["Sleep_Hours"])

            # Take the last row (the user's input after preprocessing) and predict
            X = processed.tail(1)
            X_vals = X.values  # model expects numeric array

            pred = model.predict(X_vals)
            pred_val = pred[0]

            # map numeric prediction to readable label when possible
            pred_label = readable_label(pred_val)
            st.success(f"Prediction (encoded): {pred_val}  —  Prediction (label): {pred_label}")

            # show probabilities with readable class names if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_vals)[0]
                # model.classes_ gives the class ordering used by predict_proba
                if hasattr(model, "classes_"):
                    class_names = [readable_label(c) for c in model.classes_]
                else:
                    # fallback to ordered keys from sleep_quality_map
                    class_names = [readable_label(i) for i in range(len(proba))]
                proba_df = pd.DataFrame([proba], columns=class_names)
                st.write("Probabilities (per class):")
                st.dataframe(proba_df.T.rename(columns={0: "probability"}))
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ...existing code...
def batch_prediction_page(model, df_sample):
    st.header("Batch Prediction")
    st.info("Vous pouvez uploader un CSV (raw) ou utiliser un échantillon du dataset analysé (raw).")
    use_sample = st.checkbox("Utiliser échantillon dataset analysé", value=False)
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
    df = None
    if use_sample and isinstance(df_sample, pd.DataFrame):
        df = df_sample.copy()
    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Impossible de lire le CSV uploadé: {e}")
            return
    else:
        st.warning("Aucun dataset sélectionné pour la prédiction par lot.")
        return

    # Drop 'Sleep_Quality' column immediately after loading raw data (before display / preprocessing)
        # Drop 'Sleep_Quality' column immediately after loading raw data (before display / preprocessing)
    if isinstance(df, pd.DataFrame) and "Sleep_Quality" in df.columns:
        df = df.drop(columns=["Sleep_Quality"])

    # Ensure required raw columns exist so preprocess_data won't raise KeyError
    for _col in ["ID", "Country", "Age", "Health_Issues", "Sleep_Quality"]:
        if _col not in df.columns:
            df[_col] = np.nan

    st.write("Aperçu des données (raw):")
    st.dataframe(df.head())

    # paramètres de sortie
    output_filename = st.text_input("Nom du fichier de sortie (CSV)", value="predictions.csv")
    save_on_server = st.checkbox("Enregistrer le fichier de sortie sur le serveur", value=False)

    if df is not None and st.button("Run Predictions"):
        try:
            # Prétraitement (utilise votre src.data.preprocessing.preprocess_data)
            processed = preprocess_data(df.copy())

            # Drop Sleep_Hours si présent
            if "Sleep_Hours" in processed.columns:
                processed = processed.drop(columns=["Sleep_Hours"])

            # conversion en numpy pour la prédiction
            X_vals = processed.values
            preds = model.predict(X_vals)

            # Add encoded prediction and readable label
            out = df.copy()
            out["prediction"] = preds
            try:
                out["prediction_label"] = [readable_label(p) for p in preds]
            except Exception:
                out["prediction_label"] = out["prediction"].astype(str)

            # If probabilities available, attach them with readable class columns
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_vals)
                if hasattr(model, "classes_"):
                    class_cols = [readable_label(c) for c in model.classes_]
                else:
                    class_cols = [readable_label(i) for i in range(probs.shape[1])]
                probs_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in class_cols], index=out.index)
                out = pd.concat([out, probs_df], axis=1)

            st.success("Prédictions terminées")
            st.dataframe(out.head(50))

            # préparation du CSV pour téléchargement
            csv_bytes = out.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Télécharger les prédictions (CSV)",
                data=csv_bytes,
                file_name=output_filename,
                mime="text/csv"
            )

            # optionnel : sauvegarder sur le serveur (chemin relatif au serveur)
            if save_on_server:
                try:
                    save_path = os.path.abspath(output_filename)
                    with open(save_path, "wb") as f:
                        f.write(csv_bytes)
                    st.info(f"Fichier enregistré sur le serveur : {save_path}")
                except Exception as e:
                    st.error(f"Échec de l'enregistrement sur le serveur : {e}")

        except Exception as e:
            st.error(f"Erreur lors des prédictions par lot : {e}")

def main():
    st.title("Global Coffee Health Prediction App — Dataset monitored")
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
