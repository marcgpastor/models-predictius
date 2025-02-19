import pickle

def guardar_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model guardat a: {filepath}")

def carregar_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model carregat des de: {filepath}")
    return model