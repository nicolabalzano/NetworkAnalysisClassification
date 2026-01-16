import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Impostazioni stile professionale
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.6)

# ==========================================
# 1. DATASET PREPARATION
# ==========================================
models = ['Random Forest', 'MLP Baseline', 'K-Means NN']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Dati Globali
rf_data = [0.9940, 0.9920, 0.9920, 0.9920]
mlp_data = [0.9870, 0.9871, 0.9860, 0.9865]
kmeans_data = [0.9870, 0.9862, 0.9863, 0.9862]

# Dati Per-Classe (F1-Score)
classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
rf_f1 = [1.00, 0.99, 1.00, 0.99, 0.99]
mlp_f1 = [1.00, 0.97, 1.00, 0.97, 0.99]
km_f1 = [1.00, 0.98, 0.99, 0.97, 0.99]

# ==========================================
# GRAFICO 1: COMPARISON OF GLOBAL METRICS
# ==========================================
def plot_global_metrics():
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, rf_data, width, label='Random Forest', color='#2c3e50')
    ax.bar(x, mlp_data, width, label='MLP Baseline', color='#e74c3c')
    ax.bar(x + width, kmeans_data, width, label='K-Means Augmented', color='#3498db')

    ax.set_ylabel('Score')
    ax.set_title('Global Performance Metrics Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.95, 1.0)  # Focus sulle differenze alte
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ==========================================
# GRAFICO 3: PER-CLASS F1-SCORE (Radar-style Bar Chart)
# ==========================================
def plot_per_class_f1():
    df = pd.DataFrame({
        'Class': classes * 3,
        'F1-Score': rf_f1 + mlp_f1 + km_f1,
        'Model': ['Random Forest']*5 + ['MLP Baseline']*5 + ['K-Means NN']*5
    })

    # Definire la palette di colori coerente
    palette = {'Random Forest': '#2c3e50', 'MLP Baseline': '#e74c3c', 'K-Means NN': '#3498db'}
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='F1-Score', hue='Model', data=df, palette=palette)
    plt.ylim(0.95, 1.01)
    plt.title('F1-Score per Class: Robustness on Minority Class (Class 4)', fontweight='bold')
    plt.axhline(y=0.99, color='black', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ==========================================
# ESECUZIONE
# ==========================================
plot_global_metrics()
plot_per_class_f1()
