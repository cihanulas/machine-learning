# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:13:00 2020

@author: Cihan Ulas
"""

# %%
# Load Dataset
from sklearn.datasets import load_iris
dataset = load_iris()
features = dataset.data
labels = dataset.target
labelNames = dataset.target_names
featureNames = dataset.feature_names

# print (features[:3])
# print (labels[:3])
# print (labelNames[labels[:3]])
# print (featureNames)

# %%
# Analyze Data
import pandas as pd
# print (type (features))
df_features = pd.DataFrame (features)
df_features.columns = featureNames

# print (type (df_features))
# print (df_features.describe())
# print (df_features.info())

# %% 

# Visualize Data (pandas plot) 
# To run plot Tools/IPython/Graphics de Inline to Automatic yap.
df_features.plot()  
# df_features.plot(kind="bar")  
# df_features.plot(kind="box")  
# df_features.plot(x="sepal length (cm)", y="sepal width (cm)", kind="scatter")  
# df_features.plot(x=featureNames[0], y=featureNames[1], kind="scatter")  

# %%
# Select Model: Check the road map how to scitlearn
from sklearn.neighbors import KNeighborsClassifier
knclassifier = KNeighborsClassifier(n_neighbors=10)

# %%
# Preapare Data for Model (Split Dataset with sklearn)
# import numpy as np
from sklearn.model_selection import train_test_split

X = features
Y = labels
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.33, random_state=42)
# print(len(X_train))
# print(len(X_test))


# %%
# Train Model

knclassifier.fit(X_train,Y_train)
training_accuracy = knclassifier.score (X_train,Y_train)
print ("training accuracy {:.2%}".format(training_accuracy )) 
# %%
# Test Model
test_accuracy = knclassifier.score (X_test,Y_test)
print ("test accuracy {:.2%}".format(test_accuracy ))

#%% 
# Save Model (using joblib) 
from joblib import dump, load
filename = "IrisKNNClassiferModel.joblib"
dump(knclassifier,filename);


#%% 
# Load and Test Model
clf=load(filename);
test_accuracy_loaded = clf.score (X_test,Y_test)
assert(test_accuracy_loaded==test_accuracy)
#%%