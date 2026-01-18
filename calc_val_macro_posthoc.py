import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import tensorflow as tf
from tensorflow import keras
import itertools
import warnings

# Setup come da script originale
SEED = 42
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_OUTPUT_DIR = 'grid_search_results'

def calculate_purity(labels_true, labels_pred):
    contingency_matrix = pd.crosstab(labels_pred, labels_true)
    return contingency_matrix.max(axis=1).sum() / len(labels_true)

def evaluate_kmeans_purity(X, y, k_range, random_state=SEED):
    results = {}
    print(f"Ricalcolo K-means per ricostruire split dati...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        purity = calculate_purity(y, cluster_labels)
        inertia = kmeans.inertia_
        results[k] = {'purity': purity, 'inertia': inertia, 'model': kmeans}
    return results

def get_optimal_k_elbow(results_dict):
    ks = sorted(results_dict.keys())
    purities = [results_dict[k]['purity'] for k in ks]
    p1 = np.array([ks[0], purities[0]])
    p2 = np.array([ks[-1], purities[-1]])
    max_dist = 0
    best_k = ks[0]
    for i, k in enumerate(ks):
        p0 = np.array([k, purities[i]])
        numerator = np.abs((p2[1] - p1[1]) * p0[0] - (p2[0] - p1[0]) * p0[1] + p2[0] * p1[1] - p2[1] * p1[0])
        denominator = np.linalg.norm(p2 - p1)
        dist = numerator / denominator
        if dist > max_dist:
            max_dist = dist
            best_k = k
    return best_k

def augment_dataset_with_distances(X, centroids):
    distances = cdist(X, centroids, metric='euclidean')
    X_augmented = np.hstack([X, distances])
    return X_augmented

def prepare_validation_data():
    print("[INFO] Preparazione Dati (Identica al training)...")
    try:
        df_train = pd.read_csv('cleaned_dataset.csv')
    except FileNotFoundError:
        print("Dataset non trovato!")
        return None, None

    X_train = df_train.drop(columns=['Label']).values
    y_train = df_train['Label'].values

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    n_classes = len(label_encoder.classes_)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # K fissato manuale per velocizzare
    best_k = 23
    print(f"    [INFO] Uso K fissato: {best_k}")
    
    # Addestriamo K-Means con k=23
    best_kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
    best_kmeans.fit(X_train_scaled)
    
    centroids = best_kmeans.cluster_centers_

    # Augment
    X_train_aug = augment_dataset_with_distances(X_train_scaled, centroids)

    # Shuffle (SEED 42)
    indices = np.arange(len(X_train_aug))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    X_train_aug = X_train_aug[indices]
    y_train_encoded = y_train_encoded[indices]

    # Split (SEED 42) - Qui otteniamo X_val esatto usato nel training
    _, X_val_final, _, y_val_final = train_test_split(
        X_train_aug, y_train_encoded, test_size=0.2, random_state=SEED, stratify=y_train_encoded
    )
    
    print(f"[INFO] Dati Validazione pronti. Shape: {X_val_final.shape}")
    return X_val_final, y_val_final

def update_metrics_file(model_dir, x_val, y_val):
    model_path = os.path.join(model_dir, 'model.keras')
    report_path = os.path.join(model_dir, 'report_metrics.txt')
    
    if not os.path.exists(model_path):
        return
    
    if not os.path.exists(report_path):
        return

    # Controlla se abbiamo già calcolato
    with open(report_path, 'r') as f:
        if "VALIDATION MACRO F1:" in f.read():
            return

    print(f"Processing: {model_dir}")
    try:
        model = keras.models.load_model(model_path)
        y_pred_probs = model.predict(x_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        val_macro_f1 = f1_score(y_val, y_pred, average='macro')
        
        # Scrivi nel file
        with open(report_path, 'a') as f:
            f.write(f"\nVALIDATION MACRO F1: {val_macro_f1:.4f}\n")
            
        print(f" -> Added VAL MACRO F1: {val_macro_f1:.4f}")
        
    except Exception as e:
        print(f" -> Error: {e}")

def main():
    X_val, y_val = prepare_validation_data()
    if X_val is None:
        return

    subdirs = [os.path.join(BASE_OUTPUT_DIR, d) for d in os.listdir(BASE_OUTPUT_DIR) 
               if os.path.isdir(os.path.join(BASE_OUTPUT_DIR, d))]
    
    total = len(subdirs)
    for i, subdir in enumerate(subdirs):
        print(f"[{i+1}/{total}] ", end="")
        update_metrics_file(subdir, X_val, y_val)

if __name__ == "__main__":
    main()
