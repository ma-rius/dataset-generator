from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# set print options for large arrays
np.set_printoptions(threshold=np.inf, precision=2, linewidth=np.inf)
pd.set_option('expand_frame_repr', False)

complexity_measures = [0.2, 0.4, 0.6, 0.8]
num_data_sets = 5

# define number of attributes in Test set
num_attributes = 9

data_to_plot = []

for complexity in complexity_measures:
    all_results_for_one_complexity = []
    for i in range(num_data_sets):
        print('-------- Complexity: %r --------' % complexity)
        print('-------- Iteration: %r --------' % i)
        # import assets
        # last column is target
        df = pd.read_csv('../assets/complexity_%r/data_%r.csv' % (complexity, (i+1)), sep=',', header=0)

        # prepare data for scikit learn
        y = np.array(df['label'])
        X = np.array(df[df.columns[0:num_attributes]])
        # print(X)

        # perform train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        X_sub, X_meta, y_sub, y_meta = train_test_split(X, y, test_size=0.33, random_state=42)

        # define clf
        # clf = MLPClassifier(solver=‘lbfgs’, alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        # clf = ensemble.GradientBoostingClassifier()
        # clf = SVC()
        parameters = {'loss':('deviance', 'exponential'), 'learning_rate':[0.1, 0.3,0.4,0.7,0.9]}
        # svc_parameters = {'C': [0.001, 1, 10, 100], 'kernel':['linear', 'poly']}
        clf = GridSearchCV(ensemble.GradientBoostingClassifier(), parameters, verbose=10, n_jobs=-1)
        # clf = GridSearchCV(SVC(), svc_parameters, verbose=10, n_jobs=-1)

        # start clf training
        clf.fit(X_train, y_train)

        # predict test values
        y_pred = clf.predict(X_test)

        # calculate metrics
        results = metrics.classification_report(y_test, y_pred)
        print(metrics.accuracy_score(y_test, y_pred))

        print(results)
        print(metrics.confusion_matrix(y_test, y_pred))

        all_results_for_one_complexity.append(metrics.f1_score(y_test, y_pred))

    data_to_plot.append(all_results_for_one_complexity)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.boxplot(data_to_plot, patch_artist=True)

plt.boxplot(data_to_plot, positions=complexity_measures)
# plt.xticks(complexity_measures)
# plt.show()
plt.savefig('../charts/m=%r' % num_attributes)
