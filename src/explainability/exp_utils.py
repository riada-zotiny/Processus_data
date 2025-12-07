import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from lime import lime_tabular


def calculate_probas(model, X):
    """
    Calculate reconstruction error for each sample.

    Args:
        model: used model
        X: Input data (numpy array)

    Returns:
        predictions : classe predite échantillon 
        probabilities : probabilité de l'appartenance de chaque échantillon au classes étudiée (numpy array)

    """
    # predict_proba retourne un array numpy de probabilités pour chaque classe
    probabilities = model.predict_proba(X)
    
    # Détermination de la classe prédite (indice de la proba max)
    predictions = probabilities.argmax(axis=1)

    return predictions, probabilities

def create_shap_explainer(predict_fn, data):
    """

    Args:
        predict_fn: fonction de prediction
        data: data

    Returns:
        explainer: objet SHAP KernelExplainer

    """
    return shap.Explainer(predict_fn, data[:100])


def calculate_shap_values(explainer, X):
    """
    Args:
        explainer: SHAP explainer object
        X: Données à expliquer par SHAP
        nsamples: Nombre d'échantillon

    Returns:
        shap_values: valeurs SHAP
    """
    return explainer(X)



def compare_shap_lime(shap_values, lime_weights, feature_names):
    """
    Compare SHAP and LIME explanations by checking sign agreement.

    Args:
        shap_values: SHAP values for one sample (1D numpy array)
        lime_weights: LIME weights dictionary {feature_name: weight}
        feature_names: List of feature names in order

    Returns:
        agreement_rate: Percentage of features with same sign
        disagreeing_features: List of features where SHAP and LIME disagree

    Example:
        >>> rate, features = compare_shap_lime(shap_vals, lime_dict, feature_names)
    """

    agreement_rate = 0
    disagreeing_features = []
    for i, feature in enumerate(feature_names):
        if shap_values[i] * lime_weights.get(str_in_key(feature, lime_weights)[0]) >= 0:
            agreement_rate += 1
        else:
            disagreeing_features.append(feature)
    return agreement_rate/len(feature_names), disagreeing_features




# LIME TASK 
def LimeTabularExplainer(X_train, y_train):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=sorted(y_train.unique()),   # affichage propre
        mode='classification'
    )


# Explain instance for _single_raw
def explain_instance_for_single_raw(explainer , sample , best_model):
    exp = explainer.explain_instance(
        data_row=sample,
        predict_fn=best_model.predict_proba
    )

def separe_classes(y_test):
    grouped_indices = y_test.groupby(y_test).indices
    return grouped_indices


def explain_instance_for_each_classe(grouped_indices , X_test , y_test, explainer , best_model):
    all_results = []
    for i in range(min(50, len(grouped_indices))):
        ligne = grouped_indices[0][i]
        sample = X_test.iloc[ligne]
        exp = explainer.explain_instance(sample.values, best_model.predict_proba)
    
        for feature, weight in exp.as_list():
            all_results.append({
                "index": ligne,
                "feature": feature,
                "weight": weight,
                "true_class": y_test.iloc[ligne]
            })


    df_lime = pd.DataFrame(all_results)
    return df_lime


def visualisation_lime(df_lime , classe):
    top_features = df_lime.groupby('feature')['weight'].mean().sort_values(ascending=False).head(10)
    top_features.plot(kind='barh', figsize=(8,5))
    plt.xlabel("Contribution moyenne LIME")
    plt.title("Top 10 features influençant la classe {classe}")
    plt.show()






