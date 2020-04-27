#!/usr/bin/env python
# coding: utf-8

# In[1]:


import altair as alt
import numpy as np
import pandas as pd
from pprint import pprint
#write up is in docx file


# In[2]:


from typing import Union
Data = Union[dict, pd.DataFrame]

def data_transformer(data: Data) -> Data:
    # Transform and return the data
    return transformed_data


# In[28]:


import chardet
with open(r'sportsref_download.csv', 'rb') as f:
    result = chardet.detect(f.read())
    draft07 = pd.read_csv(r'sportsref_download.csv', encoding=result['encoding'])
    
df = draft07.head(100)
df


# In[30]:


df = df.dropna()


# In[31]:


df


# In[5]:


alt.Chart(draft07).mark_point().encode(
    alt.X('PTS'),
    alt.Y('G'),
    alt.Size('Pk'),
    tooltip = [alt.Tooltip('Player'),
               alt.Tooltip('PTS'),
               alt.Tooltip('G'),
               alt.Tooltip('Pk')
              ]
).interactive()


# In[6]:


alt.Chart(draft07).mark_point().encode(
    x = "G",
    y = "PTS",
    color = "Tm",
    size = "Pk",
    tooltip=["Player", "Rk"]
).interactive()


# In[7]:


gasolplusOden = pd.read_csv("gasolpulsOden.csv")
gasolplusOden.head


# In[8]:


likely_counts = gasolplusOden['PTS']
print("Greg Oden's Carrer Point vs Kyle Korver")
print("Player - Carrer Points")
print(likely_counts)
likely_counts.plot(kind='bar', figsize=(7,5), color="green")
print("0 - Greg Oden, 1 - Kyle Korver")


# In[ ]:


import altair as alt
from vega_datasets import data

source = data.draft07()

alt.Chart(source).mark_bar().encode(
    x='Player',
    y='G',
    color='site'
)


# In[9]:


alt.Chart(draft07).mark_point().encode(
    x = "G",
    y = "PTS",
    tooltip=["Player", "Rk"]
).interactive()


# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif


# In[35]:


def runKNN(df, x, y, k): 
    # Let's assume that we're using numeric features to predict a categorical label
    X = x.values
    Y = y.values.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)
    
    # Build a kNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    
    # Compute the quality of our predictions
    score = knn.score(X_test, Y_test)
    print("Predicts " + y.name + " with " + str(score) + " accuracy")
    print("F1 score is " + str(f1_score(Y_test, knn.predict(X_test), labels=y.unique(), average='weighted')))
    print("Chance is: " + str(1.0 / len(y.unique())))
    
    return knn


# In[18]:


def loadAndCleanData(df):
    data = pd.read_csv(df)
    data = data.fillna(0)
    #print(data)
    return data


# In[38]:


df


# In[39]:


# Predict position as a function of goals (mirroring regression)
# Note that x should be a dataframe (hence the double [])
x = df[["G"]]
y = df["Yrs"]

# Run kNN with one neighbor
runKNN(df, x, y, 1)

# Run kNN with five neighbors
runKNN(df, x, y, 5)

# Measure performance for different values for k
for k in range(1, 10): 
    print(str(k) + " neighbors:")
    runKNN(df, x, y, k)
    
# Print out the number of datapoints in each class
y.value_counts()


# In[40]:


# Now let's run with a broader set of data
# We can use ALL of our numeric data to make a prediction
x = df.select_dtypes(include="number")
y = df["G"]

# Run k-NN for two different values of k
runKNN(df, x, y, 1)
runKNN(df, x, y, 5)

# Find the best k to predict position. 
# We can do this by looking at the accuracy for each setting of k
# and choosing the k that gives us the best performance
for k in range(1, 10): 
    print(str(k) + " Neighbors:")
    runKNN(df, x, y, k)


# In[33]:


df


# In[41]:


player = df[df["Player"] == "Marc Gasol"]
print(player)
playerData = player.select_dtypes(include="number")

# Build a model to use in prediction. We can see that our classifier doesn't get Ovechkin's position
# right, but is close (a LW and RW play the same general position, just on different halves of the ice)
model = runKNN(df, x, y, 7)
print("k-NN thinks " + player["Player"] + " is a " + model.predict(playerData)[0])

# We can print out the n players closest to our target (control n by changing n_neighbors). 
# Note that the players are listed in order from closest to farthest
relatedPlayers = model.kneighbors(playerData, n_neighbors=3, return_distance=False)

for p in relatedPlayers[0]: 
    print(df.iloc[int(p)]["Player"])
    
# If we repeat with Mats Zuccarello (player = df[df["Player"] =="Mats Zuccarello\zuccama01"])
# we see that just because Zuccarello looks most like Ovechkin doesn't mean Ovechkin looks 
# most like Zuccarello (though you'll notice the classifier gets Zuccarello's position right)


# In[42]:


player = df[df["Player"] == "Greg Oden"]
relatedPlayers = model.kneighbors(player.select_dtypes(include="number"), n_neighbors=5, return_distance=False)

for p in relatedPlayers[0]: 
    print(df.iloc[int(p)]["Player"])


# In[ ]:




