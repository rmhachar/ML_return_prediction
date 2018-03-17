# STEP 1: Setup

# Import packages
import re
import glob
import numpy as np
import pandas as pd
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

# Randomly assign training column
# A 1 is part of the training set, a 0 is part of the test set
df['training'] = np.random.uniform(0,1,len(df)) <= .75

data_filtered = df.drop(['Count','Company'],axis=1)

# Create the training and testing sets from data_filtered

train = data_filtered[data_filtered['training']==True]
test = data_filtered[data_filtered['training']==False]

# Print the amount of observations in each set

print('Number of observations in the training data:', len(train))

print('Number of observations in the test data:',len(test))

# STEP 3:

# Create a list of the feature column's names
features = df.drop(['Count','Company'],axis=1)

# Create an index features equal to the columns
features = features.columns[:11]

# View features
features

# Factorize the training set
y = pd.factorize(train['ten_percent_or_greater'])[0]

# STEP 4: Create and train the classifier

# Create a random forest classifier. By convention, clf means 'classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the classifier to take the training features and learn how they relate
# to the training y (whether or not a stock's annual return is ten percent or greater)
clf.fit(train[features], y)

# Apply the classifier we trained to the test data
clf.predict(test[features])[1:10]

# Create an array of predictions
predict_temp = clf.predict(test[features])

# Create a list of predictions from the array
predictions = list(predict_temp.flatten())

# Add the list to data_temp under a new column titled 'predictions'
test_data_temp = test
test_data_temp['predictions'] = predictions

# STEP 5: Create a confusion matrix to observe results

# View the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]

possible_y = np.array(['down','up'])

### Create actual english names for the plants for each predicted plant class
preds = possible_y[clf.predict(test[features])]

# Create confusion matrix
pd.crosstab(test['ten_percent_or_greater'], preds, rownames=['y'], colnames=['possible_y'])


