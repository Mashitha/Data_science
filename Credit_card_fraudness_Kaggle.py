#!/usr/bin/env python
# coding: utf-8

# In[14]:


#dataset is taken from kaggle.com
# Credit fraudness using Logistic regression model with standardising the data and without dtandardizing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


df=pd.read_csv('C:/Users/dell/Desktop/DS_Practice/Kaggle/CredtCard_fraudness/creditcard.csv')


# In[16]:


df.head()


# In[17]:


df.describe()


# In[18]:


df.isna()


# In[19]:


sns.heatmap(df.corr())


# In[25]:


sns.countplot(x='Class',data=df)


# In[26]:


sns.heatmap(df.isnull())


# In[31]:


#logistic_regression without preprocessing 
x=df.drop('Class',axis=1)
y=df['Class']


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# In[33]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[35]:


pred=lr.predict(X_test)


# In[36]:


pred


# In[42]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(pred,y_test))
print(confusion_matrix(pred,y_test))


# In[48]:


#prediction on brand new data set 
new_data =df.drop('Class',axis=1).iloc[0]
new_data


# In[54]:


new_data = new_data.values.reshape(1,-1)
#lr.predict(new_data)


# In[55]:


lr.predict(new_data)


# In[56]:


df.iloc[0]


# In[58]:


# adding preprocessing method befor training the data 
from sklearn.preprocessing import StandardScaler


# In[59]:


# Standardizing the features
df['Pamount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df['Ptime'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1,1))

#droping the time and Amount
df = df.drop(['Time','Amount'], axis = 1)
df.head()


# In[60]:


sns.heatmap(df.corr())


# In[61]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[62]:


pred=lr.predict(X_test)
pred


# In[63]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(pred,y_test))
print(confusion_matrix(pred,y_test))


# In[67]:


new_data =df.drop('Class',axis=1).iloc[0]
new_data = new_data.values.reshape(1,-1)
lr.predict(new_data)
df.iloc[0]


# In[ ]:





# In[ ]:




