"""
Modulo per la pulizia e pre-elaborazione dei dataset.

Contiene funzioni per:
- Rimuovere colonne con varianza zero
- Rimuovere colonne con valori costanti
- Rimuovere colonne con troppi valori mancanti
- Analizzare la distribuzione delle classi
- Gestire valori mancanti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def remove_zero_variance_columns(data, exclude_cols=None):
    """
    Rimuove colonne numeriche con varianza zero (tutti i valori sono uguali).
    
    Args:
        data: DataFrame pandas
        exclude_cols: Lista di colonne da escludere dall'analisi (es. 'Label')
        
    Returns:
        DataFrame pulito, lista delle colonne rimosse
    """
    if exclude_cols is None:
        exclude_cols = []
    
    data_cleaned = data.copy()
    removed_cols = []
    
    # Seleziona solo colonne numeriche
    numeric_cols = data_cleaned.select_dtypes(include=['number']).columns
    cols_to_check = [col for col in numeric_cols if col not in exclude_cols]
    
    for col in cols_to_check:
        if data_cleaned[col].std() == 0:
            print(f"Rimozione '{col}': Deviazione standard = 0 (varianza zero)")
            removed_cols.append(col)
            data_cleaned = data_cleaned.drop(columns=[col])
    
    return data_cleaned, removed_cols


def remove_constant_columns(data, exclude_cols=None):
    """
    Rimuove colonne con un solo valore unico (valore costante).
    
    Args:
        data: DataFrame pandas
        exclude_cols: Lista di colonne da escludere dall'analisi
        
    Returns:
        DataFrame pulito, lista delle colonne rimosse
    """
    if exclude_cols is None:
        exclude_cols = []
    
    data_cleaned = data.copy()
    removed_cols = []
    
    cols_to_check = [col for col in data_cleaned.columns if col not in exclude_cols]
    
    for col in cols_to_check:
        if data_cleaned[col].nunique() == 1:
            print(f"Rimozione '{col}': Valore costante (un solo valore unico)")
            removed_cols.append(col)
            data_cleaned = data_cleaned.drop(columns=[col])
    
    return data_cleaned, removed_cols


def remove_high_missing_columns(data, threshold=0.5, exclude_cols=None):
    """
    Rimuove colonne con troppi valori mancanti.
    
    Args:
        data: DataFrame pandas
        threshold: Soglia percentuale di valori mancanti (default 0.5 = 50%)
        exclude_cols: Lista di colonne da escludere dall'analisi
        
    Returns:
        DataFrame pulito, lista delle colonne rimosse
    """
    if exclude_cols is None:
        exclude_cols = []
    
    data_cleaned = data.copy()
    removed_cols = []
    
    cols_to_check = [col for col in data_cleaned.columns if col not in exclude_cols]
    
    for col in cols_to_check:
        missing_ratio = data_cleaned[col].isnull().sum() / len(data_cleaned)
        if missing_ratio > threshold:
            print(f"Rimozione '{col}': Troppi valori mancanti ({missing_ratio*100:.2f}%)")
            removed_cols.append(col)
            data_cleaned = data_cleaned.drop(columns=[col])
    
    return data_cleaned, removed_cols


def clean_dataset(data, target_column='Label', missing_threshold=0.5, verbose=True):
    """
    Funzione completa per pulire il dataset applicando tutte le operazioni di pulizia.
    
    Args:
        data: DataFrame pandas
        target_column: Nome della colonna target/classe (da escludere dalla pulizia)
        missing_threshold: Soglia per rimozione colonne con valori mancanti
        verbose: Se True, stampa informazioni dettagliate
        
    Returns:
        DataFrame pulito, dizionario con tutte le colonne rimosse per categoria
    """
    if verbose:
        print("="*60)
        print("PULIZIA DEL DATASET")
        print("="*60)
        print(f"Forma dataset originale: {data.shape}")
    
    data_cleaned = data.copy()
    all_removed = {
        'zero_variance': [],
        'constant': [],
        'high_missing': []
    }
    
    # 1. Rimuovi colonne con varianza zero
    if verbose:
        print("\n--- Rimozione colonne con varianza zero ---")
    data_cleaned, removed = remove_zero_variance_columns(
        data_cleaned, 
        exclude_cols=[target_column]
    )
    all_removed['zero_variance'] = removed
    
    # 2. Rimuovi colonne con valori costanti
    if verbose:
        print("\n--- Rimozione colonne con valori costanti ---")
    data_cleaned, removed = remove_constant_columns(
        data_cleaned, 
        exclude_cols=[target_column]
    )
    all_removed['constant'] = removed
    
    # 3. Rimuovi colonne con troppi valori mancanti
    if verbose:
        print("\n--- Rimozione colonne con troppi valori mancanti ---")
    data_cleaned, removed = remove_high_missing_columns(
        data_cleaned, 
        threshold=missing_threshold,
        exclude_cols=[target_column]
    )
    all_removed['high_missing'] = removed
    
    # Totale colonne rimosse
    total_removed = sum(len(v) for v in all_removed.values())
    
    if verbose:
        print("\n" + "="*60)
        print("RIEPILOGO PULIZIA")
        print("="*60)
        print(f"Forma dataset originale: {data.shape}")
        print(f"Forma dataset pulito: {data_cleaned.shape}")
        print(f"Colonne rimosse totali: {total_removed}")
        print(f"  - Varianza zero: {len(all_removed['zero_variance'])}")
        print(f"  - Valori costanti: {len(all_removed['constant'])}")
        print(f"  - Troppi valori mancanti: {len(all_removed['high_missing'])}")
        print("="*60)
    
    return data_cleaned, all_removed


def handle_missing_values(data, strategy='mean', columns=None):
    """
    Gestisce i valori mancanti nel dataset.
    
    Args:
        data: DataFrame pandas
        strategy: Strategia per gestire i valori mancanti
                  - 'mean': Sostituisce con la media (solo numeriche)
                  - 'median': Sostituisce con la mediana (solo numeriche)
                  - 'mode': Sostituisce con la moda
                  - 'drop': Rimuove le righe con valori mancanti
                  - 'zero': Sostituisce con 0
        columns: Lista di colonne specifiche da processare (None = tutte)
        
    Returns:
        DataFrame con valori mancanti gestiti
    """
    data_cleaned = data.copy()
    
    if columns is None:
        columns = data_cleaned.columns
    
    if strategy == 'drop':
        data_cleaned = data_cleaned.dropna(subset=columns)
        print(f"Righe rimosse: {len(data) - len(data_cleaned)}")
    
    elif strategy in ['mean', 'median']:
        numeric_cols = data_cleaned[columns].select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if data_cleaned[col].isnull().any():
                if strategy == 'mean':
                    fill_value = data_cleaned[col].mean()
                else:
                    fill_value = data_cleaned[col].median()
                data_cleaned[col] = data_cleaned[col].fillna(fill_value)
                print(f"Colonna '{col}': Valori mancanti sostituiti con {strategy} ({fill_value:.4f})")
    
    elif strategy == 'mode':
        for col in columns:
            if data_cleaned[col].isnull().any():
                fill_value = data_cleaned[col].mode()[0]
                data_cleaned[col] = data_cleaned[col].fillna(fill_value)
                print(f"Colonna '{col}': Valori mancanti sostituiti con moda ({fill_value})")
    
    elif strategy == 'zero':
        for col in columns:
            if data_cleaned[col].isnull().any():
                data_cleaned[col] = data_cleaned[col].fillna(0)
                print(f"Colonna '{col}': Valori mancanti sostituiti con 0")
    
    return data_cleaned


def analyze_class_distribution(data, class_column='Label', plot=True, save_path=None):
    """
    Analizza la distribuzione delle classi nel dataset.
    
    Args:
        data: DataFrame pandas
        class_column: Nome della colonna con le classi
        plot: Se True, genera un grafico
        save_path: Percorso dove salvare il grafico (None = mostra solo)
        
    Returns:
        DataFrame con conteggi e percentuali delle classi
    """
    print(f"\n=== DISTRIBUZIONE DELLE CLASSI: {class_column} ===\n")
    
    # Conta le occorrenze di ogni classe
    class_counts = data[class_column].value_counts().sort_index()
    class_percentages = (class_counts / len(data) * 100).round(2)
    
    # Crea un DataFrame con i risultati
    distribution = pd.DataFrame({
        'Classe': class_counts.index,
        'Conteggio': class_counts.values,
        'Percentuale': class_percentages.values
    })
    
    print(distribution.to_string(index=False))
    print(f"\nTotale campioni: {len(data)}")
    print(f"Numero di classi: {data[class_column].nunique()}")
    
    # Controlla il bilanciamento
    max_pct = class_percentages.max()
    min_pct = class_percentages.min()
    if max_pct / min_pct > 2:
        print(f"\n⚠️  ATTENZIONE: Dataset sbilanciato (rapporto max/min: {max_pct/min_pct:.2f})")
    else:
        print("\n✓ Dataset relativamente bilanciato")
    
    # Grafico
    if plot:
        plt.figure(figsize=(10, 6))
        class_counts.plot(kind='bar', color='steelblue', edgecolor='black', alpha=0.8)
        plt.title(f'Distribuzione delle Classi - {class_column}', fontsize=16, fontweight='bold')
        plt.xlabel('Classe', fontsize=12)
        plt.ylabel('Frequenza', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nGrafico salvato in: {save_path}")
        else:
            plt.show()
    
    return distribution


def get_dataset_info(data, target_column='Label'):
    """
    Restituisce informazioni dettagliate sul dataset.
    
    Args:
        data: DataFrame pandas
        target_column: Nome della colonna target
        
    Returns:
        Dizionario con informazioni sul dataset
    """
    info = {
        'n_samples': len(data),
        'n_features': len(data.columns) - 1,  # Escluso il target
        'n_classes': data[target_column].nunique() if target_column in data.columns else None,
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum(),
        'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
        'numeric_features': len(data.select_dtypes(include=['number']).columns),
        'categorical_features': len(data.select_dtypes(include=['object', 'category']).columns)
    }
    
    print("\n=== INFORMAZIONI SUL DATASET ===")
    print(f"Numero di campioni: {info['n_samples']}")
    print(f"Numero di feature: {info['n_features']}")
    if info['n_classes']:
        print(f"Numero di classi: {info['n_classes']}")
    print(f"Valori mancanti totali: {info['missing_values']}")
    print(f"Righe duplicate: {info['duplicate_rows']}")
    print(f"Uso memoria: {info['memory_usage']:.2f} MB")
    print(f"Feature numeriche: {info['numeric_features']}")
    print(f"Feature categoriche: {info['categorical_features']}")
    
    return info


def save_cleaned_dataset(data, filepath, include_index=False):
    """
    Salva il dataset pulito in un file CSV.
    
    Args:
        data: DataFrame pandas
        filepath: Percorso del file di output
        include_index: Se includere l'indice nel file CSV
    """
    data.to_csv(filepath, index=include_index)
    print(f"\n✓ Dataset pulito salvato in: {filepath}")
    print(f"  Dimensioni: {data.shape}")


# Esempio di utilizzo
if __name__ == "__main__":
    # Carica il dataset
    print("Caricamento del dataset...")
    df = pd.read_csv('dataset.csv')
    
    # Pulisci il dataset
    df_cleaned, removed_cols = clean_dataset(
        df, 
        target_column='Label',
        missing_threshold=0.5,
        verbose=True
    )
    
    # Gestisci valori mancanti (se presenti)
    missing_count = df_cleaned.isnull().sum().sum()
    if missing_count > 0:
        print(f"\nGestione di {missing_count} valori mancanti...")
        df_cleaned = handle_missing_values(df_cleaned, strategy='mean')
    
    # Analizza distribuzione delle classi
    class_dist = analyze_class_distribution(
        df_cleaned, 
        class_column='Label',
        plot=True,
        save_path='class_distribution.png'
    )
    
    # Ottieni informazioni sul dataset pulito
    info = get_dataset_info(df_cleaned, target_column='Label')
    
    # Salva il dataset pulito
    save_cleaned_dataset(df_cleaned, 'cleaned_dataset.csv')
    
    # Applica le stesse rimozioni al test dataset
    print("\n" + "="*60)
    print("PULIZIA DEL TEST DATASET")
    print("="*60)
    
    # Carica il test dataset
    df_test = pd.read_csv('test_dataset.csv')
    print(f"Forma test dataset originale: {df_test.shape}")
    
    # Raccogli tutte le colonne rimosse dal training set
    all_removed_cols = (
        removed_cols['zero_variance'] + 
        removed_cols['constant'] + 
        removed_cols['high_missing']
    )
    
    # Rimuovi le stesse colonne dal test set (se esistono)
    cols_to_remove = [col for col in all_removed_cols if col in df_test.columns]
    df_test_cleaned = df_test.drop(columns=cols_to_remove)
    
    print(f"Colonne rimosse dal test set: {len(cols_to_remove)}")
    if cols_to_remove:
        print(f"  Colonne: {cols_to_remove}")
    
    # Gestisci valori mancanti nel test set (se presenti)
    missing_count_test = df_test_cleaned.isnull().sum().sum()
    if missing_count_test > 0:
        print(f"\nGestione di {missing_count_test} valori mancanti nel test set...")
        df_test_cleaned = handle_missing_values(df_test_cleaned, strategy='mean')
    
    # Salva il test dataset pulito
    save_cleaned_dataset(df_test_cleaned, 'cleaned_test_dataset.csv')
    
    print(f"Forma test dataset pulito: {df_test_cleaned.shape}")
    
    print("\n✓ Pulizia completata con successo!")
