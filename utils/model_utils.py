import json
import os
from pathlib import Path
from tensorflow.keras.models import load_model

def save_model_and_history(model, history, model_path='model', history_path=None):
    """
    Guarda el modelo en formato H5 y el historial de entrenamiento en JSON.
    Por defecto, guarda los archivos en una carpeta "models" dentro del directorio actual.
    """
    base_dir = Path(__file__).resolve().parent.parent / "models"
    base_dir.mkdir(parents=True, exist_ok=True)

    model_filename = f"{model_path}.h5" if not model_path.endswith('.h5') else model_path
    model_full_path = base_dir / model_filename

    if history_path is None:
        history_filename = model_filename.replace('.h5', '_history.json')
    else:
        history_filename = history_path
    history_full_path = base_dir / history_filename

    model.save(model_full_path)
    print(f"Modelo guardado en: {model_full_path}")

    with open(history_full_path, 'w') as f:
        json.dump(history.history, f)
    print(f"Historial guardado en: {history_full_path}")

def load_history(history_path):
    with open(history_path, 'r') as f:
        return json.load(f)

def load_model_and_history(model_name='model'):
    """
    Carga un modelo .h5 y su historial asociado .json desde la carpeta models.
    """
    base_dir = Path(__file__).resolve().parent.parent / "models"

    model_path = base_dir / f"{model_name}.h5"
    history_path = base_dir / f"{model_name}_history.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"Historial no encontrado: {history_path}")

    model = load_model(model_path)
    with open(history_path, 'r') as f:
        history = json.load(f)

    print(f"Modelo y historial cargados desde 'models/{model_name}'")
    return model, history


def save_test_results(model_name: str, test_loss: float, test_acc: float, folder: str = 'models'):
    os.makedirs(folder, exist_ok=True)
    results_path = os.path.join(folder, f"{model_name}_test_metrics.json")
    data = {
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }
    with open(results_path, 'w') as f:
        json.dump(data, f)
    print(f"Test metrics saved to: {results_path}")

def load_test_results(model_name: str, folder: str = 'models'):
    results_path = os.path.join(folder, f"{model_name}_test_metrics.json")
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data['test_loss'], data['test_accuracy']


