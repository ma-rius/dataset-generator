from numpy import genfromtxt
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

X1, y1 = make_classification(n_features=4, random_state=0)
# define number of attributes in Test set
num_attributes = 5

# import assets
# last column is target
df = pd.read_csv('../assets/data_0.2.csv', sep=',', header=0)

# prepare data for scikit learn
y = np.array(df['label'])
X = np.array(df[df.columns[0:num_attributes-1]])

# perform train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_sub, X_meta, y_sub, y_meta = train_test_split(X, y, test_size=0.33, random_state=42)

# define clf
# clf = MLPClassifier(solver=‘lbfgs’, alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf = ensemble.GradientBoostingClassifier()
# clf = LinearSVC()
# parameters = {'loss':('deviance', 'exponential'), 'learning_rate':[0.1, 0.3,0.4,0.7,0.9]}
# clf = GridSearchCV(ensemble.GradientBoostingClassifier(), parameters, verbose=10)

# start clf training
clf.fit(X_train, y_train)

# predict test values
y_pred = clf.predict(X_test)

# calculate metrics
results = metrics.classification_report(y_test, y_pred)
print(metrics.accuracy_score(y_test, y_pred))

print(results)
print(metrics.confusion_matrix(y_test, y_pred))
