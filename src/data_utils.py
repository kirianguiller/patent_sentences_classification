import json
import os
from typing import Any, Dict, List, Literal
import numpy as np
from numpy.typing import NDArray
from pathlib import Path


PATH_LABELS_TXT = Path(__file__).parent.parent / "data" / "labels.txt"
PATH_LABELS_JSON = Path(__file__).parent.parent / "data" / "labels.json"

PATH_EMBEDDINGS_FOLDER = Path(__file__).parent.parent / "data" / "embeddings"

def convert_to_labels_dict():
    labels_dict = {}
    current_patent = "NONE"
    with open(PATH_LABELS_TXT, "r") as infile:
        for line in infile:
            line = line.strip().rstrip()
            if not line:
                print("KKK not line !!")
            if "_____" in line:
                current_patent = line.split("_____")[0]
                continue

            labels_dict[current_patent] = labels_dict.get(current_patent, []) + [line]

    with open(PATH_LABELS_JSON, "w") as outfile:
        outfile.write(json.dumps(labels_dict))

def load_labels_per_patent() -> Dict[str, List[str]]:
    with open(PATH_LABELS_JSON, "r") as infile:
        labels_per_patent: Dict[str, List[str]] = json.loads(infile.read())
    return labels_per_patent

def load_inputs_per_patent(source) -> Dict[str, NDArray]:
    inputs_per_patent: Dict[str, NDArray] = {}
    path_embeddings_source_folder = PATH_EMBEDDINGS_FOLDER / source
    for filename in os.listdir(path_embeddings_source_folder):
        patent_name = filename.split("_")[0]
        inputs_per_patent[patent_name] = np.load(os.path.join(path_embeddings_source_folder, filename))
    return inputs_per_patent

def flatten_labels_per_patent(labels_per_patent: Dict[str, List[str]]):
    patents_names = list(labels_per_patent.keys())
    patents_names.sort()
    
    labels_flattened = []
    for patent_name in patents_names:
        labels_flattened += labels_per_patent[patent_name]
    return labels_flattened

def flatten_inputs_per_patent(inputs_per_patent: Dict[str, NDArray]):
    patents_names = list(inputs_per_patent.keys())
    patents_names.sort()
    
    arrays = []
    for patent_name in patents_names:
        arrays.append(inputs_per_patent[patent_name])
    return np.concatenate(arrays)

def compute_l2i_and_i2l(labels):
    l2i = {}
    i2l = {}
    for label in labels:
        if label not in l2i:
            idx = len(l2i.keys())
            l2i[label] = idx
            i2l[idx] = label
    return l2i, i2l

def get_encoded_labels(labels):
    l2i, _ = compute_l2i_and_i2l(labels)
    
    encoded_labels = []
    for label in labels:
        encoded_labels.append(l2i[label])
    return np.array(encoded_labels)


def load_data(source: Literal["roberta", "bert4patent"]):
    labels_per_patent = load_labels_per_patent()
    inputs_per_patent = load_inputs_per_patent(source)
    
    labels_flattened = flatten_labels_per_patent(labels_per_patent)
    inputs_flattened = flatten_inputs_per_patent(inputs_per_patent)

    encoded_labels = get_encoded_labels(labels_flattened)
    return inputs_flattened, encoded_labels


class QuatentPatentSentenceDataset:
    def __init__(self, source: Literal["roberta", "bert4patent"]):
        self.source = source
        self.data = load_data(source)
        l2i, i2l = self._compute_l2i_and_i2l()
        self.labels = list(l2i.keys())

    def _compute_l2i_and_i2l(self):
        labels_per_patent = load_labels_per_patent()
        labels_flattened = flatten_labels_per_patent(labels_per_patent)
        return compute_l2i_and_i2l(labels_flattened)



if __name__ == "__main__":
    for source in ["roberta", "bert4patent"]:
        labels_per_patent = load_labels_per_patent()
        inputs_per_patent = load_inputs_per_patent(source)
        assert len(labels_per_patent.keys()) == len(inputs_per_patent.keys())
        total_samples = 0
        for a_patent in list(labels_per_patent.keys()):
            # check, for all patent, that we have same length in input and output (labels)
            assert a_patent
            assert inputs_per_patent.get(a_patent) is not None
            assert labels_per_patent.get(a_patent) is not None
            assert inputs_per_patent.get(a_patent).shape[0] == len(labels_per_patent.get(a_patent))
            total_samples += len(labels_per_patent.get(a_patent))

        # check that flattened inputs and labels are same length
        labels_flattened = flatten_labels_per_patent(labels_per_patent)
        inputs_flattened = flatten_inputs_per_patent(inputs_per_patent)
        assert len(labels_flattened) == inputs_flattened.shape[0]
        assert len(labels_flattened) == total_samples

        # check one encoder 
        encoded_labels = get_encoded_labels(labels_flattened)
        assert len(set(encoded_labels)) == len(set(labels_flattened))
        l2i, i2l = compute_l2i_and_i2l(labels_flattened)
        for label in labels_flattened:
            assert label == i2l[l2i[label]]
        print(l2i)
        
        # check whole pipeline
        X, y = load_data(source)
        assert X is not None
        assert y
        print(X.shape)
        print(len(y))

        print(y[:15])
        print(labels_flattened[:15])