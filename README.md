# HSM CLASSIFIER

This project aims at building Machine Learning Algorithms for a particular Structural Health Monitoring problem. The dataset was obtained from a laboratory three-story structure, described in the ".pdf" doc uploaded to this repository.

There are two colab notebooks. The "HSM(OriginalDataset).ipynb" works with the original dataset, where the features are time series (8192 samples) values for the 5 different channels, resulting in 40960 features. The "HSM.ipynb" works with a modified dataset, where the features are the first four statistical moments of the measured data (time series) for each channel, a total of 20 features.

The first notebook ("HSM(OriginalDataset).ipynb") was uploaded to this repository only to ilustrate the erroneous way of dealing with time series in a ML Classifier.

"Because no explicit mathematical equation can be written for the time histories produced by a random phenomenon, such as the measured data in this report, statistical procedures must be used to define the properties of the data."
