import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def find_optimal_threshold(y_true, y_prob):
    """Finds the optimal probability threshold for a classifier based on F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (recalls * precisions) / (recalls + precisions)
    
    # Convert potential NaNs (from 0/0) to 0 to prevent argmax from failing.
    f1_scores = np.nan_to_num(f1_scores)
    
    if len(f1_scores) > 0:
        best_f1_idx = np.argmax(f1_scores)
        # thresholds array can be one element shorter than f1_scores
        optimal_threshold_idx = min(best_f1_idx, len(thresholds) - 1)
        optimal_threshold = thresholds[optimal_threshold_idx]
        best_f1 = f1_scores[best_f1_idx]
        print(f"Found optimal threshold: {optimal_threshold:.4f} (F1-score: {best_f1:.4f})")
    else:
        print("Warning: Could not determine optimal threshold. Defaulting to 0.5.")
        optimal_threshold = 0.5
        
    return optimal_threshold

def evaluate_champion_model(model, X_test, y_test, device, optimal_threshold, artifact_path="artifacts"):
    print(f"\n--- Evaluating Champion Model on Test Set (using threshold: {optimal_threshold:.4f}) ---")
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test.values))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    model.eval()
    y_test_probs_list = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            y_test_probs_list.extend(probs.cpu().numpy())
    
    y_prob = np.array(y_test_probs_list).flatten()
    y_pred = (y_prob > optimal_threshold).astype(int)
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=["Bad Loan (0)", "Good Loan (1)"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.title("Confusion Matrix");
    cm_path = f"{artifact_path}/confusion_matrix.png"; plt.savefig(cm_path); plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob); roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5)); plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})'); plt.plot([0, 1], [0, 1], 'k--'); plt.title('ROC Curve'); plt.legend();
    roc_path = f"{artifact_path}/roc_curve.png"; plt.savefig(roc_path); plt.close()
    
    precision, recall, _ = precision_recall_curve(y_test, 1 - y_prob, pos_label=0); pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5)); plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})'); plt.title('PR Curve (Bad Loan)'); plt.legend();
    pr_path = f"{artifact_path}/pr_curve.png"; plt.savefig(pr_path); plt.close()
    
    return {"test_roc_auc": roc_auc, "test_pr_auc": pr_auc, "cm_path": cm_path, "roc_path": roc_path, "pr_path": pr_path}