import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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
model = IsolationForest(n_estimators=10)

# Fit the classifier
model.fit(X_train)

# Making predictions
result = model.predict(X_test_1)
result2 = model.predict(X_test_2)

# Remapping the labels
result[result == 1] = 0
result[result == -1] = 1

result2[result2 == 1] = 0
result2[result2 == -1] = 1

# Calculating the metrics
accuracy, precision, recall = [], [], []

accuracy.append(accuracy_score(result, Y_test_1))
precision.append(precision_score(result, Y_test_1))
recall.append(recall_score(result, Y_test_1))

accuracy.append(accuracy_score(result2, Y_test_2))
precision.append(precision_score(result2, Y_test_2))
recall.append(recall_score(result2, Y_test_2))

df = pd.DataFrame({"Accuracy": accuracy, "Precision": precision, "Recall": recall})
print(df)

# Plotting the graph
df.plot(xlabel = "Test Dataset", ylabel = "Performance", kind = 'bar', title = "Metrics")
plt.legend(loc = 1)
plt.xticks(rotation=0, ha='right')
plt.tight_layout()
plt.show()