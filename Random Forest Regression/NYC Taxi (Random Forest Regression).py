#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime as dt


# In[2]:


df = pd.read_csv("dataset.csv", index_col = "timestamp", parse_dates=True)
df.head()


# In[3]:


df.drop("Unnamed: 0", axis=1, inplace = True)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df['value-1']=df['value'].shift(+1)
df['value-2']=df['value'].shift(+2)
df['value-3']=df['value'].shift(+3)


# In[11]:


df.head()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df=df.dropna()
df


# In[12]:


df.isnull().sum()


# In[13]:


sns.boxplot(x=df["value"])


# In[14]:


filter1 = (df>30000)
filter1.value_counts()


# In[16]:


x1,x2,x3 = df['value-1'],df['value-2'],df['value-3']
y = df['value']
x1,x2,x3,y = np.array(x1),np.array(x2),np.array(x3),np.array(y)
x1,x2,x3,y = x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1),y.reshape(-1,1)
final_x = np.concatenate((x1,x2,x3), axis = 1)
print(final_x)


# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(final_x,y, test_size=0.2)


# In[18]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100, random_state=23, max_features=3)


# In[19]:


rfr.fit(x_train, y_train)


# In[20]:


y_pred = rfr.predict(x_test)
y_pred


# In[25]:


pred = rfr.predict(x_test)
import matplotlib.pyplot as plt
plt.plot(y_pred, label = "Regression Prediction")
plt.plot(y_test, label = "Actual Values")
plt.legend(loc="upper right")
plt.show()


# In[34]:


diff = y_pred - pred

plt.scatter(range(len(diff)), diff)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Ground Truth - Prediction')
plt.xlabel('Sample index')
plt.ylabel('Difference')
plt.grid(True)

plt.show()


# In[22]:


from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
score

