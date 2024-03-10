import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

dataset = pd.read_table("ddos_train_oc_benign.netflow",delimiter='\t')
dataset2 = pd.read_table("ts2-ddos-test01.netflow",delimiter='\t')


X_train = dataset.iloc[:,:]
X_test = dataset2.iloc[:,1:]
Y_test = dataset2.iloc[:,0:1]

# Normalizing training value for better numerical and graphical representation
ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)
X_test = ss_train.fit_transform(X_test)

# Defining the model
model = IsolationForest(n_estimators=1)


# Fit the classifier
model.fit(X_train)

# Making predictions
result = model.predict(X_test)
accuracy, precision, recall = [], [], []

print(result)
'''
accuracy.append(accuracy_score(result, Y_test))
precision.append(precision_score(result, Y_test))
recall.append(recall_score(result, Y_test))

df = pd.DataFrame({"Accuracy": accuracy, "Precision": precision, "Recall": recall}, index = dataset)
print(df)

# Plotting the graph
df.plot(xlabel = "Dataset", ylabel = "Performance", kind = 'bar', title = "Metrics")
plt.legend(loc = 1)
plt.xticks(rotation=0, ha='right')
plt.tight_layout()
plt.show()
'''