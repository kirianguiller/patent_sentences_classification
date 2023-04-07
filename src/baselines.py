from sklearn import svm
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from statistics import mean, stdev

import numpy as np

from data_utils import load_data, load_data_splitted

scores = {}
n_iter = 5
for i in range(n_iter):
    print("iteration", i + 1)
    for source in ["roberta", "bert4patent"]:
        X_train, y_train, X_test, y_test, l2i = load_data_splitted(source)

        pca = PCA(n_components=500)

        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)

        perceptron_model = Perceptron()
        perceptron_model.fit(X_train_reduced, y_train)

        score = perceptron_model.score(X_test_reduced, y_test)
        score = balanced_accuracy_score(y_test, perceptron_model.predict(X_test_reduced))
        # score = cohen_kappa_score(y_test, perceptron_model.predict(X_test_reduced))

        source_and_method_name = source + "_" + "sklearn.Perceptron()"
        scores[source_and_method_name] = scores.get(source_and_method_name, []) + [score]
        # SGDClassifier

        clf = SGDClassifier()
        clf.fit(X_train_reduced, y_train)

        score = clf.score(X_test_reduced, y_test)
        score = balanced_accuracy_score(y_test, clf.predict(X_test_reduced))
        # score = cohen_kappa_score(y_test, clf.predict(X_test_reduced))
        source_and_method_name = source + "_" + "sklearn.SGDClassifier()"
        scores[source_and_method_name] = scores.get(source_and_method_name, []) + [score]

print(f"## Results ({n_iter} iterations)")
print("|method and source|mean|std|")
print("| --- | --- | ---|")
for source_and_method_name, source_and_method_scores in scores.items():
    mean_ = round(mean(source_and_method_scores), 3)
    stdev_ = round(stdev(source_and_method_scores), 3)
    print(f"|{source_and_method_name}|{mean_}|+/- {stdev_}|")


