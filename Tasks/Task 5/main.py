import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score, precision_score, recall_score


dataset = pd.read_table("ndp-ddos-train-oc-benign.netflow",delimiter='\t')
dataset2 = pd.read_table("ndp-ddos-test01.netflow",delimiter='\t')
dataset3 = pd.read_table("ndp-ddos-test02.netflow",delimiter='\t')
dataset4 = pd.read_table("ndp-ddos-test03.netflow",delimiter='\t')

# Processing the data
X_train = dataset.iloc[:,:]
X_test_1 = dataset2.iloc[:,1:]
Y_test_1 = dataset2.iloc[:,:1]
X_test_2 = dataset3.iloc[:,1:]
Y_test_2 = dataset3.iloc[:,:1]
X_test_3 = dataset4.iloc[:,1:]
Y_test_3 = dataset4.iloc[:,:1]

# Normalizing training value for better numerical and graphical representation
ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)
X_test_1 = ss_train.fit_transform(X_test_1)
X_test_2 = ss_train.fit_transform(X_test_2)
X_test_3 = ss_train.fit_transform(X_test_3)

# Defining the model
# Contamination score was found using the graph
modelElliptical = EllipticEnvelope(contamination = 0.46)

# Fit the classifier
modelElliptical.fit(X_train)

# Making predictions
resultElliptical = modelElliptical.predict(X_test_1)
resultElliptical2 = modelElliptical.predict(X_test_2)
resultElliptical3 = modelElliptical.predict(X_test_3)

# Remapping the labels
resultElliptical[resultElliptical == 1] = 0
resultElliptical[resultElliptical == -1] = 1

resultElliptical2[resultElliptical2 == 1] = 0
resultElliptical2[resultElliptical2 == -1] = 1

resultElliptical3[resultElliptical3 == 1] = 0
resultElliptical3[resultElliptical3 == -1] = 1

# Calculating the metrics
accuracy, precision, recall = [], [], []
index = ["Elliptical1", "Elliptical2", "Elliptical3"]

accuracy.append(accuracy_score(resultElliptical, Y_test_1))
precision.append(precision_score(resultElliptical, Y_test_1))
recall.append(recall_score(resultElliptical, Y_test_1))

accuracy.append(accuracy_score(resultElliptical2, Y_test_2))
precision.append(precision_score(resultElliptical2, Y_test_2))
recall.append(recall_score(resultElliptical2, Y_test_2))

accuracy.append(accuracy_score(resultElliptical3, Y_test_3))
precision.append(precision_score(resultElliptical3, Y_test_3))
recall.append(recall_score(resultElliptical3, Y_test_3))

df = pd.DataFrame({"Accuracy": accuracy, "Precision": precision, "Recall": recall}, index = index)
print(df)

# Plotting the graph
df.plot(xlabel = "Test Dataset", ylabel = "Performance", kind = 'bar', title = "Metrics")
plt.legend(loc = 1)
plt.xticks(rotation=0, ha='center')
plt.tight_layout()
plt.show()