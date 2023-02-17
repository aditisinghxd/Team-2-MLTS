#!/usr/bin/env python
# coding: utf-8

# In[180]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime as dt


# In[181]:


df = pd.read_csv("wind_dataset.csv", index_col="DATE", parse_dates=True)
df.tail()


# In[182]:


df.shape


# In[183]:


df.info()


# In[184]:


df.isnull().sum()


# In[185]:


colu = ["IND.1","T.MAX" ,"IND.2" ,"T.MIN" ,"T.MIN.G" ]
for i in colu:
    print(i)
    print(df[i].mode())


# In[186]:


FilterInd1 = 0.0
FilterTmax = 10.0
FilterInd =0.0
FilterTMIN = 9.0
FilterTMIN_g = 5.0
df["IND.1"].fillna(FilterInd1 , inplace = True)
df["T.MAX"].fillna(FilterTmax , inplace = True)
df["IND.2"].fillna(FilterInd , inplace = True)
df["T.MIN"].fillna(FilterTMIN , inplace = True)
df["T.MIN.G"].fillna(FilterTMIN_g , inplace = True)


# In[187]:


df.isnull().sum()


# In[190]:


sns.boxplot(x=df['WIND'])


# In[189]:


z_scores = (df['WIND'] - df['WIND'].mean()) / df['WIND'].std()
outliers = np.abs(z_scores) > 3
median = df['WIND'].median()
df.loc[outliers, 'WIND'] = median

z_scores = (df['IND'] - df['IND'].mean()) / df['IND'].std()
outliers = np.abs(z_scores) > 3
median = df['IND'].median()
df.loc[outliers, 'IND'] = median

z_scores = (df['RAIN'] - df['RAIN'].mean()) / df['RAIN'].std()
outliers = np.abs(z_scores) > 3
median = df['RAIN'].median()
df.loc[outliers, 'RAIN'] = median

z_scores = (df['IND.1'] - df['IND.1'].mean()) / df['IND.1'].std()
outliers = np.abs(z_scores) > 3
median = df['IND.1'].median()
df.loc[outliers, 'IND.1'] = median

z_scores = (df['T.MAX'] - df['T.MAX'].mean()) / df['T.MAX'].std()
outliers = np.abs(z_scores) > 3
median = df['T.MAX'].median()
df.loc[outliers, 'T.MAX'] = median

z_scores = (df['IND.2'] - df['IND.2'].mean()) / df['IND.2'].std()
outliers = np.abs(z_scores) > 3
median = df['IND.2'].median()
df.loc[outliers, 'IND.2'] = median

z_scores = (df['T.MIN'] - df['T.MIN'].mean()) / df['T.MIN'].std()
outliers = np.abs(z_scores) > 3
median = df['T.MIN'].median()
df.loc[outliers, 'T.MIN'] = median

z_scores = (df['T.MIN.G'] - df['T.MIN.G'].mean()) / df['T.MIN.G'].std()
outliers = np.abs(z_scores) > 3
median = df['T.MIN.G'].median()
df.loc[outliers, 'T.MIN.G'] = median



# In[192]:


df.head()


# In[193]:


x1,x2,x3,x4,x5,x6,x7 = df['IND'], df['RAIN'],df['IND.1'],df['T.MAX'],df['IND.2'],df['T.MIN'],df['T.MIN.G']
y = df['WIND']
x1,x2,x3,x4,x5,x6,x7,y = np.array(x1),np.array(x2),np.array(x3),np.array(x4),np.array(x5),np.array(x6),np.array(x7),np.array(y)
x1,x2,x3,x4,x5,x6,x7,y = x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1),x4.reshape(-1,1),x5.reshape(-1,1),x6.reshape(-1,1),x7.reshape(-1,1),y.reshape(-1,1)
final_x = np.concatenate((x1,x2,x3,x4,x5,x6,x7), axis = 1)
print(final_x)


# In[194]:


len(final_x)


# In[195]:


len(y)


# In[196]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(final_x, y, test_size=0.2,random_state=23 ,shuffle=False)


# In[197]:


print(x_train)


# In[198]:


print(x_test)


# In[199]:


print(y_train)


# In[200]:


print(y_test)


# In[201]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100, random_state=20,bootstrap= True, max_features=7)


# In[202]:


rfr.fit(x_train, y_train)


# In[203]:


y_pred = rfr.predict(x_test)
print(y_pred)


# In[204]:


y_pred_ver = y_pred.reshape(len(y_pred), 1)
y_pred_ver = np.round(y_pred_ver)
print(y_pred_ver)


# In[205]:


y_true_ver = y_test.reshape(len(y_test), 1)
y_true_ver = np.round(y_true_ver)
print(y_true_ver)


# In[206]:


pred_comp = np.concatenate((y_true_ver, y_pred_ver), axis = 1)
print(pred_comp)


# In[207]:


df = pd.DataFrame(pred_comp)
df.head(5)


# In[208]:


pred = rfr.predict(x_test)
import matplotlib.pyplot as plt
plt.plot(y_pred, label = "Regression Prediction")
plt.plot(y_test, label = "Actual Values")
plt.legend(loc="upper left")
plt.show()


# In[209]:


from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(y_test, y_pred)
error

