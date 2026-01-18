import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import callbacks
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import itertools
import os

# read dataset
df = pd.read_csv('cleaned_dataset.csv')

print(df.shape)

# split dataset into features and labels
X = df.drop('Label', axis=1)
y = df['Label']

print(X.shape)
print(y.shape)

# split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# Load external test dataset
df_test = pd.read_csv('cleaned_test_dataset.csv')
X_ext_test = df_test.drop('Label', axis=1)
y_ext_test = df_test['Label']
print(f"External Test shape: {X_ext_test.shape}")

# scale the dataset
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_ext_test = scaler.transform(X_ext_test)

def savePlotLoss(history, d, save_path):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(f'Training and validation loss for {d} features')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()

def savePlotAccuracy(history, d, save_path):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title(f'Training and validation accuracy for {d} features')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'accuracy_plot.png'))
    plt.close()

class MetricsCallback(callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        X_val, y_val = self.validation_data
        
        # Predict
        y_pred_probs = self.model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Convert y_val from one-hot to labels if necessary
        if y_val.ndim == 2:
             y_val_labels = np.argmax(y_val, axis=1)
        else:
             y_val_labels = y_val
        
        # Calculate scores
        val_f1 = f1_score(y_val_labels, y_pred, average='weighted', zero_division=0)
        val_macro_f1 = f1_score(y_val_labels, y_pred, average='macro', zero_division=0)
        val_prec = precision_score(y_val_labels, y_pred, average='weighted', zero_division=0)
        val_rec = recall_score(y_val_labels, y_pred, average='weighted', zero_division=0)
        
        # Append to logs
        logs['val_f1'] = val_f1
        logs['val_macro_f1'] = val_macro_f1
        logs['val_precision'] = val_prec
        logs['val_recall'] = val_rec

def MLP_architecture(train_X, learning_rate=0.001, dropout_rate=0.3):
    n_cols = train_X.shape[1]
    input_layer = Input(shape=(n_cols,))  # -1 perché l'ultima colonna è la label
    x = Dense(128, activation='relu', kernel_initializer="glorot_uniform", name="l1")(input_layer)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu', kernel_initializer="glorot_uniform", name="l2")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu', kernel_initializer="glorot_uniform", name="l3")(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(5, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare data for training
y_train_cat = to_categorical(y_train, num_classes=5)
y_val_cat = to_categorical(y_val, num_classes=5)
y_ext_test_cat = to_categorical(y_ext_test, num_classes=5)

# Grid Search Parameters
param_grid = {
    'batch_size': [32, 64, 128],
    'learning_rate': [0.0001, 0.001, 0.01],
    'dropout_rate': [0.2, 0.3, 0.4],
    'epochs': [100],
    'patience': [15],
}

BASE_OUTPUT_DIR = 'grid_search_results_deep_learnb'
if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Total combinations: {len(combinations)}")

for i, params in enumerate(combinations):
    folder_name = f"BS_{params['batch_size']}_LR_{params['learning_rate']}_DO_{params['dropout_rate']}_EP_{params['epochs']}_PAT_{params['patience']}"
    current_dir = os.path.join(BASE_OUTPUT_DIR, folder_name)

    print(f"\nProcessing {i+1}/{len(combinations)}: {folder_name}")

    """if os.path.exists(current_dir):
        print("Folder exists, skipping...")
        continue"""
    
    os.makedirs(current_dir)

    model = MLP_architecture(X_train, learning_rate=params['learning_rate'], dropout_rate=params['dropout_rate'])

    es = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=params['patience'],
        min_delta=0.0001,
        restore_best_weights=True
    )
    
    metrics_cb = MetricsCallback(validation_data=(X_val, y_val_cat))

    history = model.fit(
        X_train, y_train_cat, 
        epochs=params['epochs'], 
        batch_size=params['batch_size'], 
        validation_data=(X_val, y_val_cat),
        callbacks=[es, metrics_cb], 
        verbose=0, 
        shuffle=True
    )

    # Save plots
    savePlotLoss(history, X_train.shape[1], current_dir)
    savePlotAccuracy(history, X_train.shape[1], current_dir)

    # Predizioni per metriche
    y_pred_probs = model.predict(X_ext_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calcolo Metriche
    acc = accuracy_score(y_ext_test, y_pred)
    loss = model.evaluate(X_ext_test, y_ext_test_cat, verbose=0)[0]
    
    f1 = f1_score(y_ext_test, y_pred, average='macro')
    prec = precision_score(y_ext_test, y_pred, average='macro')
    rec = recall_score(y_ext_test, y_pred, average='macro')
    
    # Confusion Matrix
    cm = confusion_matrix(y_ext_test, y_pred)
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
        f.write(f"VALIDATION MACRO F1: {history.history['val_macro_f1']}\n")
        f.write(f"Validation Precision: {history.history['val_precision']}\n")
        f.write(f"Validation Recall: {history.history['val_recall']}\n")
        f.write(f"Validation Loss Min: {min(history.history['val_loss']):.4f}\n")
        f.write(f"Validation Accuracy Max: {max(history.history['val_accuracy']):.4f}\n")
        f.write(f"Validation F1 Max: {max(history.history['val_f1']):.4f}\n")
        f.write(f"VALIDATION MACRO F1 Max: {max(history.history['val_macro_f1']):.4f}\n")
        
        f.write("\n--- MODEL CONFIGURATION ---\n")
        f.write(f"PATIENCE: {params['patience']}\n")
        f.write(f"MAX EPOCHS: {params['epochs']}\n")
        f.write(f"EPOCHS RUN: {es.stopped_epoch + 1 if es.stopped_epoch > 0 else params['epochs']}\n\n")
        
        f.write("--- CLASSIFICATION REPORT ---\n")
        f.write(classification_report(y_ext_test, y_pred))
    
    # Salva modello
    model.save(os.path.join(current_dir, 'model.keras'))

    print("Done.")
