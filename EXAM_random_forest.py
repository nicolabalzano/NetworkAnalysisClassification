from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import joblib

def determineRFkFoldConfiguration(X, y, folds=5, seed=42):
    """
    Determine the best configuration for Random Forest with respect to:
    - Criterion: gini or entropy
    - Max features: sqrt, log2, None
    - Max samples: ranging from 0.5 to 1.0 with step 0.1
    Using stratified k-fold cross-validation, based on F1 score.
    
    Returns:
        - Best criterion
        - Best max_features
        - Best max_samples
        - Average F1 score of the best configuration
        - Trained Random Forest model on the full dataset
    """
    # Hyperparameter grid
    criteria = ['gini', 'entropy']
    max_features_options = ['sqrt', 'log2', None]
    max_samples_options = np.arange(0.5, 1.01, 0.1)  # Da 0.5 a 1.0 incluso
    
    # Directory to save results
    base_results_dir = "rf_grid_search_results"
    os.makedirs(base_results_dir, exist_ok=True)
    
    best_config = None
    best_f1 = -1
    best_metrics = None # To store all metrics for the best config
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    
    # Iterate over all hyperparameter combinations
    for criterion in criteria:
        for max_features in max_features_options:
            for max_samples in max_samples_options:
                
                # Setup results directory
                config_dir_name = f"RF_CRIT_{criterion}_MF_{max_features}_MS_{max_samples:.2f}"
                current_dir = os.path.join(base_results_dir, config_dir_name)
                os.makedirs(current_dir, exist_ok=True)
                print(f"Processing configuration: {config_dir_name}")

                # Lists to store metrics for each fold
                f1_scores = []
                acc_scores = []
                prec_scores = []
                rec_scores = []
                
                # To aggregate predictions for confusion matrix
                y_true_all = []
                y_pred_all = []
                
                # Perform Stratified K-Fold Cross Validation
                for train_idx, test_idx in skf.split(X, y):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Train Random Forest
                    model = RandomForestClassifier(
                        criterion=criterion,
                        max_features=max_features,
                        max_samples=max_samples,
                        random_state=seed,
                    )
                    model.fit(X_train, y_train)
                    
                    # Evaluate on test set (validation fold)
                    y_pred = model.predict(X_test)
                    
                    f1_fold = f1_score(y_test, y_pred, average='macro')
                    #print(f"Criterion: {criterion}, Max Features: {max_features}, Max Samples: {max_samples:.2f} -> Fold F1: {f1_fold:.4f}")
                    
                    # Calculate metrics
                    f1_scores.append(f1_fold)
                    acc_scores.append(accuracy_score(y_test, y_pred))
                    prec_scores.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
                    rec_scores.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
                    
                    y_true_all.extend(y_test)
                    y_pred_all.extend(y_pred)
                
                # Calculate average metrics for this configuration
                avg_f1 = np.mean(f1_scores)
                avg_acc = np.mean(acc_scores)
                avg_prec = np.mean(prec_scores)
                avg_rec = np.mean(rec_scores)
                
                # Update best configuration if current F1 is better
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_config = (criterion, max_features, max_samples)
                    best_metrics = (avg_acc, avg_prec, avg_rec, avg_f1)
                
                # --- Save Results ---
                
                # Confusion Matrix
                """
                cm = confusion_matrix(y_true_all, y_pred_all)
                plt.figure(figsize=(8, 6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f'Confusion Matrix\nAcc: {avg_acc:.4f} | F1: {avg_f1:.4f}')
                plt.colorbar()
                
                # Add numbers in cells
                thresh = cm.max() / 2.
                for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                plt.savefig(os.path.join(current_dir, 'confusion_matrix.png'))
                plt.close()
                """
                
                # Report Testuale
                with open(os.path.join(current_dir, 'report_metrics.txt'), 'w') as f:
                    params = {
                        'criterion': criterion,
                        'max_features': max_features,
                        'max_samples': max_samples
                    }
                    f.write(f"PARAMS: {params}\n")
                    f.write(f"ACCURACY:  {avg_acc:.4f}\n")
                    f.write(f"F1 SCORE:  {avg_f1:.4f}\n")
                    f.write(f"PRECISION: {avg_prec:.4f}\n")
                    f.write(f"RECALL:    {avg_rec:.4f}\n")
                    
                    f.write("\n--- MODEL CONFIGURATION ---\n")
                    f.write(f"CRITERION: {criterion}\n")
                    f.write(f"MAX FEATURES: {max_features}\n")
                    f.write(f"MAX SAMPLES: {max_samples}\n")
                    
                    f.write("\n--- CLASSIFICATION REPORT ---\n")
                    # Note: Using aggregated predictions from K-Fold
                    f.write(classification_report(y_true_all, y_pred_all))
                
                # Save Model (User requested) - Saving the model of the last fold
                joblib.dump(model, os.path.join(current_dir, 'model.joblib'))
    
    # Train the final model on the full dataset with the best configuration
    best_criterion, best_max_features, best_max_samples = best_config
    final_model = RandomForestClassifier(
        criterion=best_criterion,
        max_features=best_max_features,
        max_samples=best_max_samples,
        random_state=seed,
        verbose=1
    )
    final_model.fit(X, y)
    
    print(f"Best Configuration: Criterion={best_criterion}, Max Features={best_max_features}, Max Samples={best_max_samples}")
    print(f"Best Validation Metrics (Avg over {folds} folds):")
    print(f"  Accuracy:  {best_metrics[0]:.4f}")
    print(f"  Precision: {best_metrics[1]:.4f}")
    print(f"  Recall:    {best_metrics[2]:.4f}")
    print(f"  F1 Score:  {best_metrics[3]:.4f}")
    
    return best_criterion, best_max_features, best_max_samples, best_f1, final_model



if "__main__" == __name__:
    import pandas as pd
    
    # Load cleaned dataset
    df = pd.read_csv('cleaned_dataset.csv')
    
    # Separate features and labels for training
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # Load external test dataset
    df_test = pd.read_csv('cleaned_test_dataset.csv')
    X_ext_test = df_test.drop('Label', axis=1)
    y_ext_test = df_test['Label']
    
    # Determine best Random Forest configuration using Stratified K-Fold
    best_criterion, best_max_features, best_max_samples, best_f1, final_model = determineRFkFoldConfiguration(X, y)
    
    # --- Final Evaluation on External Test Dataset ---
    print("\n--- Evaluating Final Model on External Test Dataset ---")
    y_pred_ext = final_model.predict(X_ext_test)
    
    acc_ext = accuracy_score(y_ext_test, y_pred_ext)
    f1_ext = f1_score(y_ext_test, y_pred_ext, average='weighted') # weighted is better for imbalanced
    prec_ext = precision_score(y_ext_test, y_pred_ext, average='weighted', zero_division=0)
    rec_ext = recall_score(y_ext_test, y_pred_ext, average='weighted', zero_division=0)
    
    print(f"External Test Set Results:")
    print(f"  Accuracy:  {acc_ext:.4f}")
    print(f"  F1 Score:  {f1_ext:.4f}")
    print(f"  Precision: {prec_ext:.4f}")
    print(f"  Recall:    {rec_ext:.4f}")
    
    # Confusion Matrix for External Test Set
    cm_ext = confusion_matrix(y_ext_test, y_pred_ext)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_ext, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'External Test Confusion Matrix\nAcc: {acc_ext:.4f} | F1: {f1_ext:.4f}')
    plt.colorbar()
    
    # Add numbers
    thresh = cm_ext.max() / 2.
    for i, j in itertools.product(range(cm_ext.shape[0]), range(cm_ext.shape[1])):
        plt.text(j, i, format(cm_ext[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm_ext[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('final_external_confusion_matrix.png')
    plt.close()
    
    # Save Classification Report
    with open('final_external_report.txt', 'w') as f:
        f.write("--- FINAL MODEL EVALUATION ON EXTERNAL TEST SET ---\n")
        f.write(f"Accuracy:  {acc_ext:.4f}\n")
        f.write(f"F1 Score:  {f1_ext:.4f}\n")
        f.write(f"Precision: {prec_ext:.4f}\n")
        f.write(f"Recall:    {rec_ext:.4f}\n\n")
        f.write("--- CLASSIFICATION REPORT ---\n")
        f.write(classification_report(y_ext_test, y_pred_ext))
        
    # save the final model
    import joblib
    joblib.dump(final_model, 'final_random_forest_model.pkl')

    