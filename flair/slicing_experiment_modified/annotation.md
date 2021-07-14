# Annotation of the Slicing Experiment

## First experiment: training only slice 1

Directory: `./1`

Expected file: 

- `./1/conll_03/eng.train` -> slice 1

Expected Lengths: 55938

Actual Lengths: 55938

## Second experiment: using slice 1 training model on predicting slice 2

Directory: `./predict_2`

Expected file: 

- `./predict_2/conll_03/eng.train` -> `dev.tsv` of slice 1 from the first experiment
- `./predict_2/conll_03/eng.testb` -> slice 2

Expected Lengths: 

- dev.tsv of slice 1: 55044
- slice 2: 53865

Actual Lengths:

- dev.tsv of slice 1: ***55938*** **(Might be the problem)**
- eng.testb: 53865

## Third experiment: Using slice 1 + predicted labels of slice 2 to train model

Directory: `./1_and_predict_2`

Expected file:

- `./1_and_predict_2/conll_03/eng.train` -> slice 1 + `dev.tsv` of the second experiment

Expected Lengths: 55044 + 55938 = 110982

Actual Lengths: 110983

## Last experiment: Using slice 1 + slice 2 to train model

Directory: `./1_and_2`

Expected file:

- `./1_and_predict_2/conll_03/eng.train` -> slice 1 + slice 2

Expected Lengths: 55938 + 53865 = 109803

Actual Lengths: 109804