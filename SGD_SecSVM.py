import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split


def getSample(data, targets, sample_size):
    np.random.seed(2019)
    random_index = np.random.choice(data.shape[0], sample_size, replace=False)
    X_sample = np.take(data, random_index, 0)
    y_sample = targets[random_index]
    return X_sample, y_sample


if __name__ == '__main__':
    feature_of_counts = "./feature_vectors_counts.csv"
    numberOfSampleForTrainingTheModule = 4000
    learning_rate = 0.02
    C = 1.0
    TotalLoop = 500

    dataset = pd.read_csv(feature_of_counts, index_col=0)
    dataset['malware'] = dataset['malware'].map({False: -1, True: 1})
    x = dataset.iloc[:, 1:9].values
    y = dataset.iloc[:, 9].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    train_feature_count = X_train.shape[1]
    train_sample_count = X_train.shape[0]
    test_feature_count = X_test.shape[1]
    test_sample_count = X_test.shape[0]

    All_weights = np.random.normal(0.0, 0.1, train_feature_count)

    for i in range(TotalLoop):
        X_sample, y_sample = getSample(X_train, y_train, numberOfSampleForTrainingTheModule)
        n = X_sample.shape[0]
        m = X_sample.shape[1]
        hinge = np.zeros((m,))
        for index, sample in enumerate(X_sample):
            wx = np.dot(sample, All_weights)
            hinge = hinge + (
                (y_sample[index]) * np.transpose(sample).reshape(m, ) if y_sample[index] * (wx) < 1 else 0)
        bias = np.concatenate((np.zeros(1), All_weights[1:m]), axis=0)
        gradient = bias - (C) / float(n) * hinge
        updated_wight = []
        for g, weight in zip(gradient, All_weights):
            weight = weight - learning_rate * g
            updated_wight.append(weight)
        All_weights = np.asarray(updated_wight)

    y_out = np.zeros((test_sample_count,))
    for index, sample in enumerate(X_test):
        if sample.dot(All_weights) >= 0:
            y_out[index] = 1
        if sample.dot(All_weights) < 0:
            y_out[index] = -1

    print("Accuracy is: ", metrics.accuracy_score(y_test, y_out) * 100)
