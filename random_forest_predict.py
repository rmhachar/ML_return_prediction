
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
from sklearn.ensemble import RandomForestClassifier

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

# View features
features

# Factorize the training set
y = pd.factorize(data_filtered['ten_percent_or_greater'])[0]
x = data_filtered

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.25, random_state =0)

print('Number of observations in the training data:', len(x_train))

print('Number of observations in the test data:',len(x_test))

data_filtered

# STEP 3: Create and Train the Classifier

# Create a random forest classifier.
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the classifier to take the training features and learn how they relate
# to the training y (whether or not a stock's annual return is ten percent or greater)
clf.fit(x_train[features], y_train)

# STEP 4: Predict using the Classifier

# Apply the Classifier we trained to the test data
clf.predict(x_test[features])[1:10]

# Create an array of predictions
predict_temp = clf.predict(x_test[features])

# Create a list of predictions from the array
predictions = list(predict_temp.flatten())

# Add the list to data_temp under a new column titled 'predictions'
test_data_temp = x_test
test_data_temp['predictions'] = predictions

# STEP 5: Create a Confusion Matrix

# View the predicted probabilities of the first 10 observations
clf.predict_proba(x_test[features])[0:10]

possible_y = np.array(['down','up'])

# Create array for predictions 'preds'
preds = possible_y[clf.predict(x_test[features])]

# Create confusion matrix
pd.crosstab(y_test, preds, rownames=['y'], colnames=['possible_y'])


# STEP 6: Feature Analysis

mask = (test_data_temp['predictions'] == 1)
test_data_temp[mask].head()
(test_data_temp[mask]).count()

cluster_data = test_data_temp[mask]
all_data = test_data_temp

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

pca = PCA(n_components=2)
PCA_reduced_df = pca.fit(scaled_data).transform(scaled_data)

features = pd.DataFrame(list(zip(cluster_data.columns, pca.components_[0], np.mean(cluster_data), np.mean(all_data))),
        columns=['Feature', 'Importance', 'Cluster Average', 'Overall Average']).sort_values('Importance', ascending=False).head(10)

features.reset_index().drop('index', axis=1)
