from sklearn import svm
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path

from data_utils import QuatentPatentSentenceDataset

PATH_FIGURES_FOLDER = PATH_LABELS_TXT = Path(__file__).parent.parent / "figures"

def compute_and_save_conf_matrix(model, X_test, y_test, name, labels):
    np.set_printoptions(precision=1)
    plt.rcParams["figure.figsize"] = (15, 15)


    # Plot non-normalized confusion matrix
    titles_options = [
        ("Confusion matrix, without normalization", None, "absolute"),
        ("Normalized confusion matrix", "true", "normalized"),
    ]
    for title, normalize, shortname in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels=labels,
            cmap=plt.cm.Blues,
            normalize=normalize,
            xticks_rotation=45,
        )
        disp.ax_.set_title(name + " : " + title)
        # disp.ax_.set_xticklabels(labels, rotation = 45)
        print(title)
        print(disp.confusion_matrix)

        plt.savefig(PATH_FIGURES_FOLDER / (name + "_" + shortname + ".jpg"))
        plt.close()


for source in ["roberta", "bert4patent"]:
    dataset = QuatentPatentSentenceDataset(source)
    X_true, y_true = dataset.data

    X_train, X_test, y_train, y_test = train_test_split(X_true, y_true, test_size=0.2)

    pca = PCA(n_components=500)

    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)

    perceptron_model = Perceptron()

    perceptron_model.fit(X_train_reduced, y_train)

    compute_and_save_conf_matrix(perceptron_model, X_test_reduced, y_test, source + "_Perceptron", dataset.labels)

    # SGDClassifier

    clf = SGDClassifier()
    clf.fit(X_train_reduced, y_train)

    y_test_pred = clf.predict(X_test_reduced)

    compute_and_save_conf_matrix(perceptron_model, X_test_reduced, y_test, source + "_SGDClassifier", dataset.labels)
    