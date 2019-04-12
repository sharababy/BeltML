#BeltML

ML Models applied to data collected from vibrations sensors,temprature sensors and tacometers.

Libraries used: Scikit Learn, Numpy

## Files:
preprocess.py : for reshaping raw data into timeseries data.
combine.py : for combining ideal data and abnormal data into single file
classify.py : for calling and running models on the data.

## Current Scores:
Accuracy Score (GBT):  0.949238578680203
Accuracy Score (DTree):  0.9441624365482234
Accuracy Score (GaussianNB):  1.0

## Dataset Size:
Train shape (Rows,datapoints): (648, 56)
Test shape: (Rows,datapoints) (197, 56)

overlap = 5 points
time series length = 7 points.