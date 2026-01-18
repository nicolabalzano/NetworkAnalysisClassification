
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import itertools
import warnings
import logging

SEED = 42

# Disabilita warning non critici
warnings.filterwarnings('ignore')
# 0 = tutti i log, 1 = no info, 2 = no warning, 3 = solo errori critici
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# Configurazione Cartelle Output
BASE_OUTPUT_DIR = 'grid_search_results'
if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)

# ============================================================================
# 2. FUNZIONI DI SUPPORTO (CLUSTERING & UTILS)
# ============================================================================

def calculate_purity(labels_true, labels_pred):
    """
    Calcola la Purity: (1/N) * sum(max(cluster_i ∩ class_j))
    """
    # per ogni cluster, trova la classe più frequente
    # crosstab crea una tabella in cui le righe sono i cluster e le colonne le classi
    # ogni cella contiene il conteggio delle occorrenze
    """
    Cluster	    Gatto	Cane	Uccello
        0	    20	    5	    0
        1	    2	    30	    3
        2	    0	    0       40
    """
    contingency_matrix = pd.crosstab(labels_pred, labels_true)
    # somma delle massime occorrenze per cluster diviso il totale
    return contingency_matrix.max(axis=1).sum() / len(labels_true)

def evaluate_kmeans_purity(X, y, k_range, random_state=SEED):
    """
    Addestra K-means per diversi k e calcola la Purity.
    """
    results = {}
    print(f"Valutazione K-means in corso (Range: {k_range.start}-{k_range.stop-1})...")
    
    for k in k_range:
        # n_init='auto' o 10 per stabilità, determina il numero di inizializzazioni indipendenti
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        purity = calculate_purity(y, cluster_labels)
        inertia = kmeans.inertia_
        
        # inertia e' la somma delle distanze quadratiche dei campioni dal centroide più vicino
        # più basso è meglio, indica cluster più compatti
        results[k] = {'purity': purity, 'inertia': inertia, 'model': kmeans}
        # print(f"  k={k}: Purity={purity:.4f}")
        
    return results

def get_optimal_k_elbow(results_dict):
    """
    Trova il k ottimale calcolando la massima distanza geometrica
    dalla retta che congiunge il primo e l'ultimo punto della curva Purity.
    """
    ks = sorted(results_dict.keys())
    purities = [results_dict[k]['purity'] for k in ks] # Lista delle purity corrispondenti
    
    # Punti estremi della curva
    p1 = np.array([ks[0], purities[0]]) 
    p2 = np.array([ks[-1], purities[-1]])
    
    max_dist = 0
    best_k = ks[0]
    
    for i, k in enumerate(ks):
        p0 = np.array([k, purities[i]])
        # Formula distanza punto-retta
        """
        misura la lunghezza del segmento perpendicolare che parte dal punto p0
        e arriva alla retta definita dai punti p1 e p2.
        
        """
        numerator = np.abs((p2[1] - p1[1]) * p0[0] - (p2[0] - p1[0]) * p0[1] + p2[0] * p1[1] - p2[1] * p1[0])
        denominator = np.linalg.norm(p2 - p1) # distanza euclidea tra p1 e p2
        
        dist = numerator / denominator
        
        if dist > max_dist:
            max_dist = dist
            best_k = k
            
    return best_k

def augment_dataset_with_distances(X, centroids):
    """
    Aggiunge le distanze euclidee dai centroidi come nuove feature.
    """
    distances = cdist(X, centroids, metric='euclidean')
    distance_cols = [f'dist_centroid_{i}' for i in range(distances.shape[1])]
    X_augmented = np.hstack([X, distances])
    return X_augmented, distance_cols

# ============================================================================
# 3. FUNZIONI DI VISUALIZZAZIONE
# ============================================================================

def plot_purity_vs_k(results, best_k=None, save_dir='.'):
    k_values = sorted(list(results.keys()))
    purity_values = [results[k]['purity'] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, purity_values, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Purity')
    plt.title('Ottimizzazione di k: Analisi Purity (Elbow)')
    plt.grid(True, alpha=0.3)
    
    if best_k:
        best_purity = results[best_k]['purity']
        plt.axvline(x=best_k, color='r', linestyle='--', label=f'Elbow Point (k={best_k})')
        plt.legend()
        plt.annotate(f'k={best_k}\nP={best_purity:.4f}', 
                     xy=(best_k, best_purity), 
                     xytext=(best_k+2, best_purity-0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

    path = os.path.join(save_dir, 'purity_optimization.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    axes[0].plot(history.history['accuracy'], label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss Plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300)
    plt.close()

# ============================================================================
# 4. BUILDER RETE NEURALE & CALLBACKS
# ============================================================================

class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        X_val, y_val = self.validation_data
        
        # Predict
        y_pred_probs = self.model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate scores
        val_f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        val_prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        val_rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        
        # Append to logs
        logs['val_f1'] = val_f1
        logs['val_precision'] = val_prec
        logs['val_recall'] = val_rec
        
        # Optional: print inline
        # print(f" - val_f1: {val_f1:.4f} - val_prec: {val_prec:.4f} - val_rec: {val_rec:.4f}")

def build_neural_network(input_dim, n_classes, learning_rate=0.001, dropout_rate=0.3, first_layer_nodes=128, n_hidden_layers=3):
    model = Sequential()
    # First Hidden Layer (and Input)
    model.add(Dense(first_layer_nodes, activation='relu', input_dim=input_dim))
    
    current_nodes = first_layer_nodes
    # Additional Hidden Layers
    for _ in range(n_hidden_layers - 1):
        model.add(Dropout(dropout_rate))
        current_nodes = max(int(current_nodes / 2), n_classes + 5)
        model.add(Dense(current_nodes, activation='relu'))

    # Output Layer
    model.add(Dense(n_classes, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)

    # ------------------------------------------------------------------------
    # STEP 1: Caricamento Dati
    # ------------------------------------------------------------------------
    print("\n[1] Caricamento Dataset...")
    try:
        df_train = pd.read_csv('cleaned_dataset.csv')
        df_test = pd.read_csv('cleaned_test_dataset.csv')
    except FileNotFoundError:
        print("ERRORE: Assicurati che i file .csv siano nella stessa cartella dello script.")
        return

    # Separazione X e y
    X_train = df_train.drop(columns=['Label']).values
    y_train = df_train['Label'].values
    X_test = df_test.drop(columns=['Label']).values
    y_test = df_test['Label'].values

    # Encoding delle labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    n_classes = len(label_encoder.classes_)
    
    print(f"    Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"    Classi ({n_classes}): {label_encoder.classes_}")

    # ------------------------------------------------------------------------
    # STEP 2: Scaling (MinMax)
    # ------------------------------------------------------------------------
    print("\n[2] MinMax Scaling...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("    Scaling completato.")

    # ------------------------------------------------------------------------
    # STEP 3: Ottimizzazione K (Purity + Elbow)
    # ------------------------------------------------------------------------
    print("\n[3] Ottimizzazione Cluster K (Purity)...")
    # Range da n_classi a 50
    k_range = range(n_classes, 51)
    
    # Esegue K-means per ogni k
    kmeans_results = evaluate_kmeans_purity(X_train_scaled, y_train_encoded, k_range)
    
    # Trova il miglior k con metodo geometrico
    best_k = get_optimal_k_elbow(kmeans_results)
    best_purity = kmeans_results[best_k]['purity']
    
    print(f"    ✓ K Ottimale (Elbow): {best_k}")
    print(f"    ✓ Purity raggiunta:   {best_purity:.4f}")
    
    # Salva grafico Purity
    plot_purity_vs_k(kmeans_results, best_k=best_k, save_dir=BASE_OUTPUT_DIR)

    # ------------------------------------------------------------------------
    # STEP 4: Augmentation (Distanze Euclidee)
    # ------------------------------------------------------------------------
    print("\n[4] Dataset Augmentation (Aggiunta distanze)...")
    best_kmeans = kmeans_results[best_k]['model']
    centroids = best_kmeans.cluster_centers_
    
    # Aggiungi colonne distanza
    X_train_aug, _ = augment_dataset_with_distances(X_train_scaled, centroids)
    X_test_aug, _ = augment_dataset_with_distances(X_test_scaled, centroids)
    
    print(f"    Nuove feature aggiunte: {best_k}")
    print(f"    Nuova shape Input: {X_train_aug.shape}")
    
    # Shuffle dei dati di training
    indices = np.arange(len(X_train_aug))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    X_train_aug = X_train_aug[indices]
    y_train_encoded = y_train_encoded[indices]

    # Split Train/Val
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train_aug, y_train_encoded, test_size=0.2, random_state=SEED, stratify=y_train_encoded
    )
    print(f"    Split Train/Val: Train={X_train_final.shape}, Val={X_val_final.shape}")

    # ------------------------------------------------------------------------
    # STEP 5: Grid Search Rete Neurale
    # ------------------------------------------------------------------------
    print("\n[5] Avvio Grid Search (Hyperparameter Tuning)...")
    
    # --- GRIGLIA PARAMETRI ---
    param_grid = {
        'batch_size': [32, 64, 128],
        'learning_rate': [0.0001, 0.001, 0.01],
        'dropout_rate': [0.2, 0.3, 0.4],
        'n_hidden_layers': [2, 3],
        'first_layer_nodes': [64, 128, 256],
        'epochs': [100],
        'patience': [15],
    }
    
    """
    param_grid = {
        'batch_size': [32, 64, 128],
        'learning_rate': [0.0001, 0.001, 0.01],
        'dropout_rate': [0.2, 0.3, 0.4],
        'n_hidden_layers': [2, 3],
        'first_layer_nodes': [64, 128, 256],
        'epochs': [100],
        'patience': [15],
    }

    """
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"    Totale configurazioni da testare: {len(combinations)}")
        

    for i, params in enumerate(combinations):
        print(f"\n    --- Configurazione {i+1}/{len(combinations)}: {params} ---")
        
        # Nome cartella
        folder_name = f"BS_{params['batch_size']}_LR_{params['learning_rate']}_DO_{params['dropout_rate']}_FLN_{params['first_layer_nodes']}_EP_{params['epochs']}_PAT_{params['patience']}_HL_{params['n_hidden_layers']}"
        current_dir = os.path.join(BASE_OUTPUT_DIR, folder_name)
        
        if os.path.exists(current_dir):
            print(f"        -> Cartella già esistente. Skipping...")
            continue
            
        os.makedirs(current_dir)
        
        # Build Model
        model = build_neural_network(
            input_dim=X_train_final.shape[1],
            n_classes=n_classes,
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate'],
            first_layer_nodes=params['first_layer_nodes'],
            n_hidden_layers=params['n_hidden_layers']
        )
        
        # Callbacks
        es = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)
        metrics_cb = MetricsCallback(validation_data=(X_val_final, y_val_final))
        
        # Training
        history = model.fit(
            X_train_final, y_train_final,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val_final, y_val_final),
            callbacks=[es, metrics_cb],
            verbose=0
        )
        
        # Predizioni per metriche
        y_pred_probs = model.predict(X_test_aug)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calcolo Metriche
        acc = accuracy_score(y_test_encoded, y_pred)
        loss = model.evaluate(X_test_aug, y_test_encoded, verbose=0)[0]

        f1 = f1_score(y_test_encoded, y_pred, average='macro')
        prec = precision_score(y_test_encoded, y_pred, average='macro')
        rec = recall_score(y_test_encoded, y_pred, average='macro')
        
        # Salvataggio Risultati
        plot_training_history(history, current_dir)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix\nAcc: {acc:.4f} | F1: {f1:.4f}')
        plt.colorbar()
        
        # Aggiungi numeri nelle celle
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(os.path.join(current_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Report Testuale
        with open(os.path.join(current_dir, 'report_metrics.txt'), 'w') as f:
            f.write(f"PARAMS: {params}\n")
            f.write(f"ACCURACY:  {acc:.4f}\n")
            f.write(f"F1 SCORE:  {f1:.4f}\n")
            f.write(f"PRECISION: {prec:.4f}\n")
            f.write(f"RECALL:    {rec:.4f}\n")
            f.write(f"TEST LOSS: {loss:.4f}\n")
            f.write(f"Validation Loss: {history.history['val_loss']}\n")
            f.write(f"Validation Accuracy: {history.history['val_accuracy']}\n")
            f.write(f"Validation F1: {history.history['val_f1']}\n")
            f.write(f"Validation Precision: {history.history['val_precision']}\n")
            f.write(f"Validation Recall: {history.history['val_recall']}\n")
            f.write(f"Validation Loss Min: {min(history.history['val_loss']):.4f}\n")
            f.write(f"Validation Accuracy Max: {max(history.history['val_accuracy']):.4f}\n")
            f.write(f"Validation F1 Max: {max(history.history['val_f1']):.4f}\n")
            
            f.write("\n--- MODEL CONFIGURATION ---\n")
            f.write(f"FIRST LAYER NODES: {params['first_layer_nodes']}\n")
            f.write(f"HIDDEN LAYERS: {params['n_hidden_layers']}\n")
            f.write(f"PATIENCE: {params['patience']}\n")
            f.write(f"MAX EPOCHS: {params['epochs']}\n")
            f.write(f"EPOCHS RUN: {es.stopped_epoch if es.stopped_epoch > 0 else params['epochs']}\n\n")
            
            f.write("--- CLASSIFICATION REPORT ---\n")
            f.write(classification_report(y_test_encoded, y_pred, target_names=[str(c) for c in label_encoder.classes_]))
        
        # Salva modello
        model.save(os.path.join(current_dir, 'model.keras'))
        
        #print(f"        -> Risultati salvati. Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    print("\nGrid Search completato.")


if __name__ == "__main__":
    main()