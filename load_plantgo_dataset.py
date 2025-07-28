import numpy as np
import re

def arff_to_numpy(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        full = f.read().strip().split("@data\n")
        attributes = full[0]
        data = full[1]
        n = len(re.findall('({)', data))
        m = int(re.findall('@attribute Att(\d+) numeric', attributes)[-1])
        k = len(re.findall('@attribute Label(\d+) {0,1}', attributes))
        X = np.zeros((n, m))
        y = np.zeros((n, k))
        for i, line in enumerate(data.split("\n")):
            indices = np.array(re.findall('(\d+) 1', line)).astype(int)
            feature_indices = indices[indices < m]
            label_indices = np.mod(indices[indices >= m], m)
            X[i, feature_indices] = 1
            y[i, label_indices] = 1
    return X, y

Xtrain, ytrain = arff_to_numpy('PlantGo-train.arff')

Xtest, ytest = arff_to_numpy('PlantGo-test.arff')

print(Xtrain.shape)
print(ytrain.shape)
print(Xtest.shape)
print(ytest.shape)