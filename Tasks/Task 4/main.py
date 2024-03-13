import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score, precision_score, recall_score


dataset = pd.read_table("ddos_train_oc_benign.netflow",delimiter='\t')
dataset2 = pd.read_table("ts2-ddos-test01.netflow",delimiter='\t')
dataset3 = pd.read_table("ts2-ddos-test02.netflow",delimiter='\t')

# Processing the data
X_train = dataset.iloc[:,:]
X_test_1 = dataset2.iloc[:,1:]
Y_test_1 = dataset2.iloc[:,:1]
X_test_2 = dataset3.iloc[:,1:]
Y_test_2 = dataset3.iloc[:,:1]

# Normalizing training value for better numerical and graphical representation
ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)
X_test_1 = ss_train.fit_transform(X_test_1)
X_test_2 = ss_train.fit_transform(X_test_2)

# Defining the model
modelForest = IsolationForest(n_estimators = 10)
modelElliptical = EllipticEnvelope(contamination = 0.18)

# Fit the classifier
modelForest.fit(X_train)
modelElliptical.fit(X_train)

# Making predictions
resultForest = modelForest.predict(X_test_1)
resultForest2 = modelForest.predict(X_test_2)

resultElliptical = modelElliptical.predict(X_test_1)
resultElliptical2 = modelElliptical.predict(X_test_2)

# Remapping the labels
resultForest[resultForest == 1] = 0
resultForest[resultForest == -1] = 1

resultForest2[resultForest2 == 1] = 0
resultForest2[resultForest2 == -1] = 1

resultElliptical[resultElliptical == 1] = 0
resultElliptical[resultElliptical == -1] = 1

resultElliptical2[resultElliptical2 == 1] = 0
resultElliptical2[resultElliptical2 == -1] = 1

# Calculating the metrics
accuracy, precision, recall = [], [], []
index = ["Forest1", "Forest2", "Elliptical1", "Elliptical2"]

accuracy.append(accuracy_score(resultForest, Y_test_1))
precision.append(precision_score(resultForest, Y_test_1))
recall.append(recall_score(resultForest, Y_test_1))

accuracy.append(accuracy_score(resultForest2, Y_test_2))
precision.append(precision_score(resultForest2, Y_test_2))
recall.append(recall_score(resultForest2, Y_test_2))

accuracy.append(accuracy_score(resultElliptical, Y_test_1))
precision.append(precision_score(resultElliptical, Y_test_1))
recall.append(recall_score(resultElliptical, Y_test_1))

accuracy.append(accuracy_score(resultElliptical2, Y_test_2))
precision.append(precision_score(resultElliptical2, Y_test_2))
recall.append(recall_score(resultElliptical2, Y_test_2))

df = pd.DataFrame({"Accuracy": accuracy, "Precision": precision, "Recall": recall}, index = index)
print(df)

# Plotting the graph
df.plot(xlabel = "Test Dataset", ylabel = "Performance", kind = 'bar', title = "Metrics")
plt.legend(loc = 1)
plt.xticks(rotation=0, ha='center')
plt.tight_layout()
plt.show()