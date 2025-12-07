import json

file_uri = "../data/experiments/exps.json"


def load_runs():
    try:
        with open(file_uri, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = {}
    
    return data

def save_runs(data):

    # Sauvegarder les changements effectués
    with open(file_uri, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print(f"Le fichier a été mis à jour.")

def save_run(experiment, run_id, score, model_name):
    data = load_runs()

    # Cas de nouvelle experience
    if experiment not in data:
        data[experiment] = {}
        
    # Affectation des valeur au dictionnaire
    data[experiment][run_id] = {
        'model' : model_name, 
        'score' : score
    }
    
    save_runs(data)


def get_experiment_runs(experiment):
   
    #Prend en paramettre le nom d'une experience et renvoie les ID des run qui ont été effectuées au sein d'elle
    data = load_runs()

    return data.get(experiment, {}).keys()

def get_scores(experiment, run_id):

    #Prend en paramettre le nom d'une experience et l'id d'une run et renvoie le modèle utilisé et le score obtenue dans cette dernière
    data = load_runs()

    if experiment in data:
        if run_id in data[experiment]:
            return data[experiment][run_id]
    return None