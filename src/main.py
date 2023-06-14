import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

import models.classical_ml as tm
import models.evaluation_utils as eu

senti_labels = ["Negative", "Neutral", "Positive"]

def load_encoded_data():
    pass

def load_tfidf_data():
    input_folder_path = "./input/"
    processed_folder_path = "./input/processed"

    train = pd.read_csv(os.path.join(input_folder_path, "train.csv"))
    val = pd.read_csv(os.path.join(input_folder_path, "val.csv"))
    test = pd.read_csv(os.path.join(input_folder_path, "test.csv"))
    y_train = train['y'].values
    y_val = val['y'].values
    y_test = test['y'].values

    x_train = load_npz(os.path.join(processed_folder_path, "train_tfidf.npz"))
    x_val = load_npz(os.path.join(processed_folder_path, "val_tfidf.npz"))
    x_test = load_npz(os.path.join(processed_folder_path, "test_tfidf.npz"))

    return x_train, y_train, x_val, y_val, x_test, y_test

def main():
    x_train_tfidf, y_train, x_val_tfidf, y_val, x_test_tfidf, y_test = load_tfidf_data()
    print(x_train_tfidf)
    print(y_train)
    print(x_train_tfidf.shape, y_train.shape)
    print(x_val_tfidf.shape, y_val.shape)
    print(x_test_tfidf.shape, y_test.shape)

    '''
        NAIVE BAYES
    '''
    print("NAIVE BAYES")
    nb = tm.NaiveBayes(x_train_tfidf, y_train, x_val_tfidf, y_val, x_test_tfidf, y_test) 
    nb.perform_grid_search() # Save top 3 models

    # Fit the top 3 models, find the model among top 3 with the highest validation accuracy, and store it
    trained_models = nb.train_and_evaluate_models()
    # Evaluate and print metrics for each model in fitted models for more in-depth analysis
    results_dir = "./results/NB_results/NB_trained"
    for i, (model, params) in enumerate(trained_models):
        y_pred = model.predict(x_train_tfidf)
        print(i)
        print("-Training: ")
        eu.evaluate_model(y_pred, f"Training-NB-{i}", x_train_tfidf, y_train, params, senti_labels, results_dir, only_metrics=True)
        y_pred = model.predict(x_val_tfidf)
        print("-Validation:")
        eu.evaluate_model(y_pred, f"Validation-NB-{i}", x_val_tfidf, y_val, params, senti_labels, results_dir, only_metrics=True)

    # Use the best model to evaluate on the test set
    nb_best_model, nb_best_params = nb.evaluate_best_model()
    print(f"*Best model: {nb_best_model}")
    y_pred = nb_best_model.predict(x_test_tfidf)
    print(np.bincount(y_pred))
    results_dir = "./results/NB_results/NB_best"
    model_type = "NB-best"
    eu.evaluate_model(y_pred, model_type, x_test_tfidf, y_test, nb_best_params, senti_labels, results_dir, only_metrics=False)
    eu.calculate_OvR_roc_auc_score(nb_best_model, model_type, x_train_tfidf, y_train, x_test_tfidf, y_test, senti_labels)
    print()


    '''
        RANDOM FOREST
    '''
    print("RANDOM FOREST")
    rf = tm.RandomForest(x_train_tfidf, y_train, x_val_tfidf, y_val, x_test_tfidf, y_test) 
    rf.perform_grid_search() # Save top 3 models

    # Fit the top 3 models, find the model among top 3 with the highest validation accuracy, and store it
    trained_models = rf.train_and_evaluate_models()
    # Evaluate and print metrics for each model in fitted models for more in-depth analysis
    results_dir = "./results/RF_results/RF_trained"
    for i, (model, params) in enumerate(trained_models):
        y_pred = model.predict(x_train_tfidf)
        print(i)
        print("-Training: ")
        eu.evaluate_model(y_pred, f"Training-RF-{i}", x_train_tfidf, y_train, params, senti_labels, results_dir, only_metrics=True)
        y_pred = model.predict(x_val_tfidf)
        print("-Validation:")
        eu.evaluate_model(y_pred, f"Validation-RF-{i}", x_val_tfidf, y_val, params, senti_labels, results_dir, only_metrics=True)

    # Use the best model to evaluate on the test set
    rf_best_model, rf_best_params = rf.evaluate_best_model()
    print(f"*Best model: {nb_best_model}")
    y_pred = rf_best_model.predict(x_test_tfidf)
    print(np.bincount(y_pred))
    results_dir = "./results/NB_results/NB_best"
    model_type = "NB-best"
    eu.evaluate_model(y_pred, model_type, x_test_tfidf, y_test, rf_best_params, senti_labels, results_dir, only_metrics=False)
    eu.calculate_OvR_roc_auc_score(rf_best_model, model_type, x_train_tfidf, y_train, x_test_tfidf, y_test, senti_labels)
    #eu.plot_feature_imp(model, results_dir=results_dir)
    print()


    '''
        SUPPORT VECTOR MACHINES
    '''
    print("SVM")
    svm = tm.SVC(x_train_tfidf, y_train, x_val_tfidf, y_val, x_test_tfidf, y_test) 
    svm.perform_grid_search() # Save top 3 models

    # Fit the top 3 models, find the model among top 3 with the highest validation accuracy, and store it
    trained_models = svm.train_and_evaluate_models()
    # Evaluate and print metrics for each model in fitted models for more in-depth analysis
    results_dir = "./results/SVM_results/SVM_trained"
    for i, (model, params) in enumerate(trained_models):
        y_pred = model.predict(x_train_tfidf)
        print(i)
        print("-Training: ")
        eu.evaluate_model(y_pred, f"Training-SVM-{i}", x_train_tfidf, y_train, params, senti_labels, results_dir, only_metrics=True)
        y_pred = model.predict(x_val_tfidf)
        print("-Validation:")
        eu.evaluate_model(y_pred, f"Validation-SVM-{i}", x_val_tfidf, y_val, params, senti_labels, results_dir, only_metrics=True)

    # Use the best model to evaluate on the test set
    svm_best_model, svm_best_params = svm.evaluate_best_model()
    print(f"*Best model: {svm_best_model}")
    y_pred = svm_best_model.predict(x_test_tfidf)
    print(np.bincount(y_pred))
    results_dir = "./results/SVM_results/SVM_best"
    model_type = "SVM-best"
    eu.evaluate_model(y_pred, model_type, x_test_tfidf, y_test, svm_best_params, senti_labels, results_dir, only_metrics=False)
    eu.calculate_OvR_roc_auc_score(svm_best_model, model_type, x_train_tfidf, y_train, x_test_tfidf, y_test, senti_labels)
    print()


if __name__ == "__main__":
    main()