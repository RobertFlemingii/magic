# ⭐️ Code and Resources ⭐️
# 🔗 Supervised learning (classification/MAGIC): https://colab.research.google.com/dri...
# 🔗 Supervised learning (regression/bikes): https://colab.research.google.com/dri...
# 🔗 Unsupervised learning (seeds): https://colab.research.google.com/dri...
# 🔗 Dataets (add a note that for the bikes dataset, they may have to open the downloaded csv file and remove special characters)
# 🔗 MAGIC dataset: https://archive.ics.uci.edu/ml/datase...
# 🔗 Bikes dataset: https://archive.ics.uci.edu/ml/datase...
# 🔗 Seeds/wheat dataset: https://archive.ics.uci.edu/ml/datase...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#################################################
#     LOADING AND PREPROCESSING DATA
#################################################

# reads the data file, adds names to the columns, and can print the first five rows
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df.head()
# print(df.head())

# changes gs and hs to 0s and 1s
df["class"] = (df["class"] == "g").astype(int)

# updates and prints the data
df.head()
# print(df.head())

#################################################
#       DATA VISUALIZATION
#################################################

# plots several histograms color-coded based on the class of all the data
for label in cols[:-1]:
    plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

#################################################
#     DATA SPLITTING
#################################################

# splits the data into three groups and shuffles it to ensure it's not in order of class
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

#################################################
#       DATA SCALING AND OVERSAMPLING
#################################################

# this function scales the feature data
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # this gives the user the option to address the class imbalance
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y

# scales the three groups of data
train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

print(classification_report(y_test, y_pred))