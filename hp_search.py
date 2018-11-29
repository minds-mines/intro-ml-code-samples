"""A tutorial on hyperparameter search using GridSearchCV
"""

from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Step 1: Prepare data
iris = load_iris()
# This data is a classification dataset with 3 classes and an input of 4 features

# Step 2: Select Model
svc = SVC(random_state=0)
# We can set a particular hyperparameter for all tests by adding it to the original instantiation

# Step 3: Determine Hyperparameters to test
parameters = {'kernel': ('linear', 'rbf', "poly"), 'C': [
    1, 10], 'gamma': ["scale"]}
# We can set a particular hyperparameter for all tests by adding it as a list with 1 item

# Step 4: Determine evaluation metric

# Step 5: Run search and determine best hyperparameters
