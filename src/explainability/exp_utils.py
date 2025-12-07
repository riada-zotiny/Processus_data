import numpy as np

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
    return shap.KernelExplainer(predict_fn, data[:100])


def calculate_shap_values(explainer, X, nsamples=50):
    """
    Args:
        explainer: SHAP explainer object
        X: Données à expliquer par SHAP
        nsamples: Nombre d'échantillon

    Returns:
        shap_values: valeurs SHAP
    """
    return explainer.shap_values(X, nsamples = nsamples)


def create_lime_explainer(data, feature_names, mode='regression'):
    """
    Create a LIME TabularExplainer.

    Args:
        data: données
        feature_names: nom des variables explicatives
        mode: 'regression' or 'classification'

    Returns:
        explainer: objet LIME TabularExplainer

    """
    return lime_tabular.LimeTabularExplainer(
        training_data=data,
        feature_names=feature_names,
        mode=mode,
        verbose=False
    )

def explain_instance_lime(explainer, instance, predict_fn, num_features=14, num_samples=500):
    """
    Generate LIME explanation for a single instance.

    Args:
        explainer: LIME explainer object
        instance: Single sample to explain (1D numpy array)
        predict_fn: Function that takes X and returns predictions
        num_features: Number of features to explain
        num_samples: Number of perturbation samples

    Returns:
        explanation: LIME explanation object
        feature_weights: Dictionary of {feature_name: weight}
    """

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=num_features,
        num_samples=num_samples
    )
    feature_weights = dict(explanation.as_list())
    return explanation, feature_weights


def str_in_key(chaine, dic):
    keys = []
    for key in dic.keys():
        if chaine+" " in key:
            keys.append(key)
    return keys

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