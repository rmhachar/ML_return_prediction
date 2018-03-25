# STEP 1: Setup

# Import packages
import re
import glob
import numpy as np
import pandas as pd
import sklearn

# Import Sci-Kit Learn
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

# Set random seed
np.random.seed(0)

# Load File
file_loc = "market_return_data_full_sample.csv"

# Read File
data = pd.read_csv(file_loc)

# Create Data Frame from file
df = pd.DataFrame(data,columns=data.columns)

# STEP 2: Split DF into training set and testing set randomly

data_filtered = df.drop(['Count','Company'],axis=1)

# Create a list of the feature column's names
features = df.drop(['Count','Company'],axis=1)

# Create an index features equal to the columns
features = features.columns[:11]

# STEP 3: Factorize and split the data

# Factorize the training set
y = pd.factorize(data_filtered['ten_percent_or_greater'])[0]
x = data_filtered

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.25, random_state =0)

print('Number of observations in the training data:', len(x_train))

print('Number of observations in the test data:',len(x_test))

# STEP 4: Standardize the Data

scaler = StandardScaler()
scaled_data_train = scaler.fit_transform(x_train[features])
scaled_data_test = scaler.fit_transform(x_test[features])

# STEP 5: Create and Train the Classifier

# Create the classifier.
clf = SVC(C=3.3, gamma=.2)

# Train the classifier using .fit
clf.fit(scaled_data_train,y_train)

# STEP 6: Create a Confusion Matrix

# View the predicted probabilities of the first 10 observations
#clf.predict_proba(x_test[features])[0:10]

possible_y = np.array(['down','up'])

# Create array for predictions 'preds'
preds = possible_y[clf.predict(scaled_data_test)]

# Create confusion matrix
pd.crosstab(y_test, preds, rownames=['y'], colnames=['possible_y'])


