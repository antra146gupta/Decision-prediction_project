#!/usr/bin/env python
# coding: utf-8

# # GreyCampus DataScience Bootcamp Project by MUSKAN GUPTA 
# 

# To Do - You are hired by a venture capitalist to predict the profit of a startup. Fo that you have to deal with a dataset which contains the details of 50 startup’s and predicts the profit of a new Startup based on certain features. Based on your decision and prediction, whether one should invest in a particular startup or not.

# In[1]:


# importing libs
import numpy as np
import pandas as pd


# In[2]:


#loading data
data=pd.read_csv('C:/Users/MUSKANG/Downloads/50_Startups.csv')


# In[3]:


data.describe()


# In[4]:


data.info()
#As there are no null value and the count is also 50 for all the columns so there is no missing value as well


# In[5]:


#for selecting Feature 
features = data.iloc[:,:-1].values # independent variable
label = data.iloc[:,[-1]].values   #dependent variable


# In[6]:


features


# In[7]:


#convert the categorical features to numerical features as 
#sklearn works only with numpy array
#Instead of label enconding and then onehotencoding, 
#newer version directly works with onehotencoding using ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [3]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
features = transformer.fit_transform(features.tolist())
features


# In[8]:


#converting an object to normal array
features = features.astype(float)


# In[9]:


features


# Splitting & Training the data

# In[10]:


#sampling the dataset
#Her we took 10% for test set and remaining 90% for train set
#Training set will be used to train the model
#Create Training and Testing sets
# Testing set will be used to test our model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,
                                                label,
                                                test_size=0.1,
                                                random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Applying models

# In[11]:


#Decision tree regressor
from sklearn.tree import DecisionTreeRegressor
for i in range(3,10):
    DTR = DecisionTreeRegressor(max_depth=3)
    DTR.fit(X_train,y_train)
    print("max_depth =  ",i)
    if i % 2 == 0:
        continue
    else:
        
        print("Training Score =",DTR.score(X_train,y_train))
        print("TEsting Score = ",DTR.score(X_test,y_test))
        
#training & testing scores for even max_depth is not printed as it is generally odd 


# In[12]:


#since it is still not generalised lets take max_depth = 5
from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor(max_depth=5)
DTR.fit(X_train,y_train)


# In[13]:


#checking score of training as well as testing
print(DTR.score(X_train,y_train))
print(DTR.score(X_test,y_test))


# In[14]:


#Random forest regressor
from sklearn.ensemble import RandomForestRegressor
for i in range(3,10):
    RF=RandomForestRegressor(n_estimators=3)
    RF.fit(X_train,y_train.ravel())
    print("n_estimator = ",i)
    print("Training Score =",RF.score(X_train,y_train))
    print("TEsting Score = ",RF.score(X_test,y_test))


# You can conclude that with n_estimator 5, generalized model canbe derived where testing score is more than training score.

# In[22]:


from sklearn.ensemble import RandomForestRegressor
RF=RandomForestRegressor(n_estimators=7)
RF.fit(X_train,y_train.ravel())

print(RF.score(X_train,y_train))
print(RF.score(X_test,y_test))


# In[23]:


#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_train_predict = lin_model.predict(X_train)
print ("Training score is: ",lin_model.score(X_train,y_train))
print ("Test score is: ", lin_model.score(X_test, y_test))


# In[24]:


results = {'Decision Tree':[DTR.score(X_train,y_train), DTR.score(X_test,y_test)],
                            'Random Forest':[RF.score(X_train,y_train), RF.score(X_test,y_test)],
                            'Linear Regression':[lin_model.score(X_train,y_train), lin_model.score(X_test,y_test)]}
resultsdf = pd.DataFrame(data=results, index=["Training score", "Test score"])
testscores = resultsdf.T["Test score"]


# In[25]:


winner = testscores.idxmax()
resultsdf.style.apply(lambda x: ['background: lightblue' if x.name == winner else '' for i in x])


# In the results we can easily see that Random Forest has been selected as the optimal method for this dataset.

# # Prediction & Visualisation

# In[26]:


#Predict based on the DecisionTreeRegressor model
dtr_pred = DTR.predict(X_test)
# Compare actual and predicted values
df_dtr = pd.DataFrame({'Real Profit Values':y_test.reshape(-1), 'Predicted Profit Values':dtr_pred.reshape(-1)})
df_dtr


# In[27]:


#Predict based on the RandonForestRegressor model
rf_pred = RF.predict(X_test)
# Compare actual and predicted values
df_rf = pd.DataFrame({'Real Profit Values':y_test.reshape(-1), 'Predicted Profit Values':rf_pred.reshape(-1)})
df_rf


# In[28]:


#Predict based on linear regression model
lf_pred = lin_model.predict(X_test)
#Compare actual and predicted values
df_lf = pd.DataFrame({'Real Profit Values':y_test.reshape(-1), 'Predicted Profit Values':lf_pred.reshape(-1)})
df_lf


# In[29]:


import matplotlib.pyplot as plt
df_dtr.plot.bar(title='Decision Tree Regression',color=['royalblue','silver'])
plt.show()


# In[30]:


df_rf.plot.bar(title='Random Forrest Regression',color=['royalblue','silver'])
plt.show()


# In[31]:


df_lf.plot.bar(title='Linear Regression',color=['royalblue','silver'])
plt.show()


# In[ ]:


#thanks_for_your_time.

