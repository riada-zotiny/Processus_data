import mlflow
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import glob, os, joblib, pickle

def load_model_auto():
    """
    Cherche et charge un modèle dans l'ordre :
      1) model.pkl dans le projet
      2) tout .pkl sous ../mlruns
      3) dossier modèle MLflow (mlflow.sklearn.load_model)
    Retourne l'objet modèle ou lève FileNotFoundError.
    """
    tried = []
    # 1) cherche model.pkl à la racine ou récursivement
    candidates = glob.glob("**/model.pkl", recursive=True)
    # 2) cherche .pkl sous mlruns
    candidates += glob.glob("../mlruns/**/*.pkl", recursive=True)
    # 3) dossiers possibles MLflow
    mlflow_dirs = glob.glob("../mlruns/*/*/artifacts/*") + glob.glob("../mlruns/*/models/*")
    tried.extend(candidates + mlflow_dirs)

    # essayer les .pkl d'abord
    for path in candidates:
        try:
            # joblib peut charger les modèles sklearn
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

    # essayer les dossiers mlflow (load_model)
    for d in mlflow_dirs:
        try:
            model = mlflow.sklearn.load_model(d)
            st.write(f"Model loaded (mlflow) from: {d}")
            return model
        except Exception:
            # si dossier contient un .pkl, essayer aussi
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

def main():
    st.title("Machine Learning Prediction App")
    st.sidebar.title("Navigation")
    
    # Navigation
    page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction"])

    # Load model
    try:
        model = load_model_auto()
    except Exception as e:
        st.error(f"Model file not found. {e}")
        # afficher les fichiers mlruns pour debug
        st.write("Fichiers trouvés sous ../mlruns (exemples) :")
        st.write(glob.glob("../mlruns/**/*", recursive=True)[:20])
        return
    
    if page == "Single Prediction":
        single_prediction_page(model)
    else:
        batch_prediction_page(model)

def single_prediction_page(model):
    st.header("Single Prediction")
    st.write("Enter the feature values below:")
    
    # Create input forms (adjust features based on your dataset)
    col1, col2 = st.columns(2)
    
    with col1:
        feature1 = st.number_input("Feature 1", value=0.0)
        feature2 = st.number_input("Feature 2", value=0.0)
        feature3 = st.number_input("Feature 3", value=0.0)
    
    with col2:
        feature4 = st.number_input("Feature 4", value=0.0)
        feature5 = st.number_input("Feature 5", value=0.0)
        feature6 = st.selectbox("Feature 6 (Categorical)", ["Option A", "Option B", "Option C"])
    
    # Predict button
    if st.button("Predict"):
        # Prepare input data
        input_data = np.array([[feature1, feature2, feature3, feature4, feature5, 
                                0 if feature6 == "Option A" else 1 if feature6 == "Option B" else 2]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display result
        st.success(f"Prediction Result: {prediction[0]}")
        
        # Optional: Show probability if classifier
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_data)
            st.write("Prediction Probabilities:")
            st.write(proba)

def batch_prediction_page(model):
    st.header("Batch Prediction")
    st.write("Upload a CSV file with multiple instances for prediction")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())
        
        # Predict button
        if st.button("Run Predictions"):
            try:
                # Make predictions
                predictions = model.predict(df)
                
                # Add predictions to dataframe
                df['Prediction'] = predictions
                
                # Optional: Add probabilities
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(df.drop('Prediction', axis=1))
                    for i in range(proba.shape[1]):
                        df[f'Probability_Class_{i}'] = proba[:, i]
                
                st.success("Predictions completed!")
                st.dataframe(df)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()