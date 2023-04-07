from sklearn.model_selection import train_test_split
from .model_utils import Tagger
from .data_utils import QuatentPatentSentenceDataset

dataset = QuatentPatentSentenceDataset("bert4patent")
X_true, y_true = dataset.data
X_train, X_test, y_train, y_test = train_test_split(X_true, y_true, test_size=0.2, shuffle=False)
n_labels = len(dataset.labels)

tagger = Tagger(n_labels, X_train.shape[0])

for epoch in range(10):
