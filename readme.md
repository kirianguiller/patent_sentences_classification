# Patent Sentence Classification
## Split
The split was done uniformely, so we tried to have the same types of patent in both train and test datasets

- train_patent ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] 2156 sentences
- test_patent ['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2'] 1968 sentences


## Results (5 iterations)
|method and source|mean|std|
| --- | --- | ---|
|roberta_sklearn.Perceptron()|0.29|+/- 0.021|
|roberta_sklearn.SGDClassifier()|0.387|+/- 0.056|
|bert4patent_sklearn.Perceptron()|0.337|+/- 0.052|
|bert4patent_sklearn.SGDClassifier()|0.211|+/- 0.007|

## Confusion Matrices
### bert4patent _ 

image.png