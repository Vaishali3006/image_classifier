#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


data=pd.read_csv('mnist.csv')


# In[15]:


data.head()


# In[22]:


a=data.iloc[3,1:].values


# In[25]:


a=a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[26]:


df_x=data.iloc[:,1 :]
df_y=data.iloc[:,0]


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)


# In[28]:


y_train.head()


# In[30]:


rf=RandomForestClassifier(n_estimators=100)


# In[31]:


rf.fit(x_train,y_train)


# In[32]:


pred=rf.predict(x_test)


# In[33]:


pred


# In[45]:


s=y_test.values

count=0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count=count+1


# In[46]:


count


# In[47]:


len(pred)


# In[48]:


8080/8400

