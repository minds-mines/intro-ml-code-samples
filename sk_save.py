"""Saving and loading sklearn models
"""

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

#####################################
# Perform hyperparameter search
#####################################

# Step 1: Prepare data
iris = load_iris()
# This data is a classification dataset with 3 classes and an input of 4 features

# Step 2: Select Model
svc = SVC(random_state=0)
# We can set a particular hyperparameter for all tests by adding it to the original instantiation

# Step 3: Determine Hyperparameters to test
parameters = {'kernel': ('linear', 'rbf', "poly"), 'C': [
    0.1, 0.5, 1], 'gamma': ["scale"]}
# We can set a particular hyperparameter for all tests by adding it as a list with 1 item

# Step 4: Determine evaluation metric
metric = "accuracy"

# Step 5: Run search and determine best hyperparameters
clf = GridSearchCV(svc, parameters, cv=5, scoring=metric)
clf.fit(iris.data, iris.target)

print(f"The best model uses {clf.best_params_} which results in a(n) {metric} score of {clf.best_score_}.")

#####################################
# Model Saving
#####################################

# Step 1: Determine model to save
model = clf.best_estimator_

# Step 2: Save the model
joblib.dump(model, 'models/sk-model.joblib')

#####################################
# Model Loading
#####################################

# Step 1: Determine file location
model_file = 'models/sk-model.joblib'

# Step 2: Load model
loaded_model = joblib.load(model_file)

print(f"The saved model had params {model.get_params()}.")
print(f"The loaded model has params {loaded_model.get_params()}.")
