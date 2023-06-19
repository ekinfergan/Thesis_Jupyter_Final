import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from functools import cache


class BaseModel:
    def __init__(self, model, model_type, param_grid, x_train, y_train, x_val, y_val, x_test, y_test):
        self.model = model
        self.model_type = model_type
        self.param_grid = param_grid
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.best_model = None
        self.params = None
        self.top3_params = None
        self.best_accuracy = 0
    
    @cache
    def perform_grid_search(self):
        print("*Performing grid search...")
        # Perform grid search
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=3, n_jobs=-1, verbose=10)
        grid_search.fit(self.x_train, self.y_train)

        # Get the mean test scores and standard deviations of test scores for all parameter combinations
        results_df = pd.DataFrame(grid_search.cv_results_)
        sorted_results = results_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=[False, True])
        top3_models = sorted_results[:3]
        self.print_top3_models(top3_models)
        self.top3_params = top3_models['params'].values
    
    def print_top3_models(self, top3_models):  
        print("*Printing top 3 models...")
        # Print the sorted list of mean test scores and standard deviation of test scores
        print("Top 3 parameter combinations ranked by performance (from best to worst):")
        for index, row in top3_models.iterrows():
            mean_score = row['mean_test_score']
            std_score = row['std_test_score']
            params = row['params']
            print(f"Mean Test Score: {mean_score:.4f} (±{std_score:.4f}) for {params}")

    def train_and_evaluate_models(self):
        print("*Training and Evaluating Top 3 Models...")
        trained_models = []
        for i in range(3):
            if self.model_type == "RF":
                model = RandomForestClassifier(**self.top3_params[i])
            elif self.model_type == "NB":
                model = MultinomialNB(**self.top3_params[i])
            elif  self.model_type == "SVM":
                model = SVC(**self.top3_params[i])
            else:
                print(f"Unknown model type: {self.model_type}")
                return
            model.fit(self.x_train, self.y_train)
            # Get accuracy for the validation set (.score calls .predict() internally)
            val_accuracy = model.score(self.x_val, self.y_val)
            if val_accuracy > self.best_accuracy:
                # Store the best model
                self.best_model = model
                self.best_params = self.top3_params[i]
                self.best_accuracy = val_accuracy
                idx = i
            trained_models.append((model, self.top3_params[i]))
        
        print(f"Model {idx}-{self.best_params} gives highest validation accuracy {self.best_accuracy:.2f}%")

        # Return the fitted models and their respective params for more in-depth evaluation
        return trained_models

    def evaluate_best_model(self):
        if self.best_model is not None:
            return self.best_model, self.best_params
        else:
            print(f"No best model found for {self.model_type}.")


class RandomForest(BaseModel):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        model = RandomForestClassifier()
        # Define the parameter grid for grid search
        rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 3, 5, 7, 10],  # Limit maximum depth of the trees
        'min_samples_split': [2, 5, 10, 20],  # Higher values will prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
        'min_samples_leaf': [1, 2, 5, 10, 15],  # Higher values prevent a model from getting too complex
        }
        super().__init__(model, "RF", rf_param_grid, x_train, y_train, x_val, y_val, x_test, y_test)


class NaiveBayes(BaseModel):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        model = MultinomialNB()
        # Define the parameter grid for grid search
        nb_param_grid = {
            'alpha': [0.001, 0.01, 0.1],  # Smoothing parameter for MultinomialNB
            'fit_prior': [True, False]
        }
        super().__init__(model, "NB", nb_param_grid, x_train, y_train, x_val, y_val, x_test, y_test)


class SVM(BaseModel):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        model = SVC()
        # Define the parameter grid for grid search
        svm_param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': [0.1, 1, 'scale']
        }
        super().__init__(model, "SVM", svm_param_grid, x_train, y_train, x_val, y_val, x_test, y_test)