#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor # using package of testing VIF in statsmodels
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df_sp=pd.read_csv('/Users/xuyk/dab/assignment/S&P1500_Raw_Dataset_Data.csv')
df_sp.head()


# In[3]:


df_sp = df_sp.rename(columns={'SIC Code':'SIC','Number of Employees':'EmployeesNumber',
                              'Total Assets':'TotalAssets','R&D Expense Adjusted':'R&D_Adjusted',
                              'Environmental Disclosure Score':'EDS',"Tobin's Q Ratio":'TobinsQ',
                              'Return on Assets':'ROA','Return on Common Equity':'ROE',
                              'Gross Margin':'GrossMargin'})
df_sp.head()


# In[4]:


data1=df_sp
data2=data1[['Name','Year','SIC',
             'TotalAssets','R&D_Adjusted','EDS',
             'TobinsQ','ROA','GrossMargin']]


# In[5]:


data3=data2.loc[data1['SIC']>3499]
data4=data3.loc[data2['SIC']<3600]
df_manufacturing=data4
df_manufacturing.head()


# In[6]:


df_manufacturing.shape


# In[7]:


df_manufacturing = df_manufacturing.replace('#N/A N/A', np.NaN )
df_manufacturing = df_manufacturing.replace(' ' , np.NaN )

null_all = df_manufacturing. isnull().sum()
print(df_manufacturing.isnull().sum())


# In[9]:


df =df_manufacturing
df_manufacturing = df.dropna()


df_manufacturing['Log_TotalAssets'] = np.log(df_manufacturing['TotalAssets'])

display(df_manufacturing.head())


# In[10]:


df_manufacturing.shape


# In[11]:


df_manufacturing.describe()


# In[12]:


y = df_manufacturing.ROA
X = df_manufacturing[["EDS","R&D_Adjusted","Log_TotalAssets","GrossMargin"]].assign(const=1)

results = sm.OLS(y, X).fit()
print(results.summary())


# In[13]:


y = df_manufacturing.TobinsQ
X = df_manufacturing[["EDS","R&D_Adjusted","Log_TotalAssets","GrossMargin"]].assign(const=1)

results = sm.OLS(y, X).fit()
print(results.summary())


# In[14]:



vif = [variance_inflation_factor(exog=X.values, exog_idx=i) for i in range(X.shape[1])]

vif_table = pd.DataFrame({'coef_name': X.columns, 'vif': np.around(vif,3)})
print(vif_table)

X.corr(method = 'pearson')


# In[ ]:




