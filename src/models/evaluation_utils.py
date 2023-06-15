import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from itertools import cycle

NUM_of_CLASSES = 3

def one_hot_encode(y):
    y_encoded = np.zeros((len(y), NUM_of_CLASSES))
    for i, label in enumerate(y):
        y_encoded[i, label - 1] = 1

    return y_encoded

def calculate_metrics(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted', labels=np.unique(y_pred))

    print(f"Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, f1-score: {f1:.2f}")
    
    return accuracy, precision, recall, f1

def calculate_classification_report(y, y_pred):
    return classification_report(y, y_pred)

def plot_confusion_matrix(y_true, y_pred, labels, res_path):
    cnf_mat = confusion_matrix(y_true, y_pred)
    mat_disp = ConfusionMatrixDisplay(confusion_matrix=cnf_mat, display_labels=labels)
    mat_disp = mat_disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title(f'Confusion Matrix')
    plt.savefig(os.path.join(res_path, "confusion_matrix.png"))
    plt.close()

def plot_roc_curve(prob_test_vec, y_test, labels, res_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    labels = labels
    colors = cycle(['limegreen', 'dodgerblue', 'red'])
    for senti, color in zip(range(NUM_of_CLASSES), colors):
        RocCurveDisplay.from_predictions(
            y_test[:, senti],
            prob_test_vec[:, senti],
            name=f"ROC curve for {labels[senti]}",
            color=color,
            ax=ax,
        )
    plt.savefig(os.path.join(res_path, "roc_curve.png"))
    plt.close()
        
def calculate_OvR_roc_auc_score(model, model_name, x, y, x_test, y_test, labels, res_path): #average??
    y = one_hot_encode(y)
    y_test = one_hot_encode(y_test)

    ovr_model = OneVsRestClassifier(model).fit(x, y)
    prob_test_vec = ovr_model.predict_proba(x_test)
    
    fpr, tpr, thresholds, auc_score = [], [], [], []
    for _ in range(NUM_of_CLASSES):
        fpr.append(0)
        tpr.append(0)
        thresholds.append(0)
        auc_score.append(0)
    
    for i in range(NUM_of_CLASSES):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], prob_test_vec[:, i])
        auc_score[i] = auc(fpr[i], tpr[i])

    averaged_auc_score = (sum(auc_score) / NUM_of_CLASSES)
    # Save AUC to results.txt
    with open(os.path.join(res_path, f"{model_name}_results.txt"), "a") as f:
        f.write(f"AUC score: {auc_score}\n")
        f.write(f"Averaged AUC score: {averaged_auc_score:.2f}\n")

    plot_roc_curve(prob_test_vec, y_test, labels, res_path=res_path)


def evaluate_model(y_pred, model_name, x, y, params, labels, res_path, only_metrics):
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    with open(os.path.join(res_path, f"{model_name}_results.txt"), "w") as f:
        f.write(f"*{model_name}\n")
        f.write(f"Params: {params}\n\n")

        accuracy, precision, recall, f1 = calculate_metrics(y, y_pred)
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"f1-score: {f1:.2f}\n\n")

        if not only_metrics:
            report = calculate_classification_report(y, y_pred)
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n")

            plot_confusion_matrix(y, y_pred, labels=labels, res_path=res_path)

# TODO:
def plot_feature_imp(model, res_path):
    importances = model.feature_importances_
    feature_names = tfidf_vectorizer.get_feature_names_out()
    feature_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    feature_importances.nlargest(20).plot.bar(ax=ax)
    ax.set_title("Top 20 Most Predictive Features")
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    fig.tight_layout()
    plt.savefig(os.path.join(res_path, "feature_importance.png"))
    plt.close()
