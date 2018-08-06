# import matplotlib as plt
# plt.use('Agg')
import matplotlib.pyplot as plt
from numpy import genfromtxt
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# set print options for large arrays
np.set_printoptions(threshold=np.inf, precision=2, linewidth=np.inf)
pd.set_option('expand_frame_repr', False)

complexity_measures = [0.2, 0.4, 0.6, 0.8]
num_data_sets = 100

# define number of attributes in Test set
num_attributes = 6

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
        # clf = GaussianNB()
        # clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        # clf = KNeighborsClassifier(3)
        # parameters = {'loss':('deviance', 'exponential'), 'learning_rate':[0.1, 0.3,0.4,0.7,0.9]}
        # svc_parameters = {'C': [0.001, 1, 10, 100], 'kernel':['linear', 'poly']}
        knc_parameters = {'n_neighbors': [2, 3, 5, 7, 9, 15], 'weights': ['uniform', 'distance'],
                          'leaf_size': [20, 30, 50, 80, 200]}
        # clf = GridSearchCV(ensemble.GradientBoostingClassifier(), parameters, verbose=10, n_jobs=-1)
        # clf = GridSearchCV(SVC(), svc_parameters, verbose=10, n_jobs=-1)
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
        #                     hidden_layer_sizes=(5, 2), random_state=1)
        clf = GridSearchCV(KNeighborsClassifier(), knc_parameters, verbose=1, n_jobs=-1)

        # start clf training
        clf.fit(X_train, y_train)
        print('Best parameters:', clf.best_params_)
        # predict test values
        y_pred = clf.predict(X_test)

        # calculate metrics
        results = metrics.classification_report(y_test, y_pred)
        print(metrics.accuracy_score(y_test, y_pred))

        print(results)
        print(metrics.confusion_matrix(y_test, y_pred))

        all_results_for_one_complexity.append(metrics.f1_score(y_test, y_pred))

    data_to_plot.append(all_results_for_one_complexity)

fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
ax.boxplot(data_to_plot)
ax.set_xticklabels(complexity_measures)
# plt.show()
plt.savefig('../charts/m=%r' % num_attributes)
