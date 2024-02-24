#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing various packages
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import KNNImputer


# In[2]:


#Reading the csv file
df=pd.read_csv('F:\\Unified Mentor Internship\\Financial Analytics data.csv')


# In[3]:


#displaying top 10 rows
df.head(10)


# In[4]:


#checking how many null values are there
df.isna().sum()


# In[5]:


#creating a copy of the original dataframe
df1=df


# In[6]:


#storing the values of both these columns for KNN imputation
data = df1[['Mar Cap - Crore', 'Sales Qtr - Crore']]


# In[7]:


#initialize the KNN imputer
imputer = KNNImputer(n_neighbors=5)


# In[8]:


#impute the missing values
imputed_data = imputer.fit_transform(data)


# In[9]:


#updating the dataset with imputed values
df1[['Mar Cap - Crore', 'Sales Qtr - Crore']] = imputed_data


# In[10]:


#checking random rows of the data
df1.sample(10)


# In[11]:


#checking if the 'Mar Cap - Crore' column has any outliers
sns.boxplot(data=df1, x='Mar Cap - Crore')


# In[12]:


#checking if the 'Sales Qtr - Crore' column has any outliers
sns.boxplot(data=df1, x='Sales Qtr - Crore')


# In[13]:


#both Mar Cap - Crore and Sales Qtr - Crore columns has a lots of outliers


# In[14]:


#I will use IQR method to eliminate the outliers
#calculating first and third quartiles
Q1 = df1['Mar Cap - Crore'].quantile(0.25)
Q3 = df1['Mar Cap - Crore'].quantile(0.75)

#Calculating IQR
IQR = Q3-Q1

#defining lower and upper limits
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

#filtering the dataset to remove the outliers
df1 = df1[(df1['Mar Cap - Crore'] >= lower) & (df1['Mar Cap - Crore'] <= upper)]


# In[15]:


#checking if we still have the outliers using boxplot
sns.boxplot(data=df1, x='Mar Cap - Crore')


# In[16]:


#I will use IQR method to eliminate the outliers
#calculating first and third quartiles
Q1 = df1['Mar Cap - Crore'].quantile(0.25)
Q3 = df1['Mar Cap - Crore'].quantile(0.75)

#Calculating IQR
IQR = Q3-Q1

#defining lower and upper limits
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

#filtering the dataset to remove the outliers
df1 = df1[(df1['Mar Cap - Crore'] >= lower) & (df1['Mar Cap - Crore'] <= upper)]


# In[17]:


#checking if we still have the outliers using boxplot
sns.boxplot(data=df1, x='Mar Cap - Crore')


# In[18]:


#I will use IQR method to eliminate the outliers
#calculating first and third quartiles
Q1 = df1['Mar Cap - Crore'].quantile(0.25)
Q3 = df1['Mar Cap - Crore'].quantile(0.75)

#Calculating IQR
IQR = Q3-Q1

#defining lower and upper limits
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

#filtering the dataset to remove the outliers
df1 = df1[(df1['Mar Cap - Crore'] >= lower) & (df1['Mar Cap - Crore'] <= upper)]


# In[19]:


#checking if we still have the outliers using boxplot
sns.boxplot(data=df1, x='Mar Cap - Crore')


# In[20]:


#I will use IQR method to eliminate the outliers
#calculating first and third quartiles
Q1 = df1['Mar Cap - Crore'].quantile(0.25)
Q3 = df1['Mar Cap - Crore'].quantile(0.75)

#Calculating IQR
IQR = Q3-Q1

#defining lower and upper limits
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

#filtering the dataset to remove the outliers
df1 = df1[(df1['Mar Cap - Crore'] >= lower) & (df1['Mar Cap - Crore'] <= upper)]


# In[21]:


#checking if we still have the outliers using boxplot
sns.boxplot(data=df1, x='Mar Cap - Crore')


# In[24]:


#I will use IQR method to eliminate the outliers
#calculating first and third quartiles
Q1 = df1['Mar Cap - Crore'].quantile(0.25)
Q3 = df1['Mar Cap - Crore'].quantile(0.75)

#Calculating IQR
IQR = Q3-Q1

#defining lower and upper limits
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

#filtering the dataset to remove the outliers
df1 = df1[(df1['Mar Cap - Crore'] >= lower) & (df1['Mar Cap - Crore'] <= upper)]


# In[25]:


#checking if we still have the outliers using boxplot
sns.boxplot(data=df1, x='Mar Cap - Crore')


# In[26]:


#I will use IQR method to eliminate the outliers
#calculating first and third quartiles
Q1 = df1['Mar Cap - Crore'].quantile(0.25)
Q3 = df1['Mar Cap - Crore'].quantile(0.75)

#Calculating IQR
IQR = Q3-Q1

#defining lower and upper limits
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

#filtering the dataset to remove the outliers
df1 = df1[(df1['Mar Cap - Crore'] >= lower) & (df1['Mar Cap - Crore'] <= upper)]


# In[27]:


#checking if we still have the outliers using boxplot
sns.boxplot(data=df1, x='Mar Cap - Crore')


# In[28]:


#I will use IQR method to eliminate the outliers
#calculating first and third quartiles
Q1 = df1['Mar Cap - Crore'].quantile(0.25)
Q3 = df1['Mar Cap - Crore'].quantile(0.75)

#Calculating IQR
IQR = Q3-Q1

#defining lower and upper limits
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

#filtering the dataset to remove the outliers
df1 = df1[(df1['Mar Cap - Crore'] >= lower) & (df1['Mar Cap - Crore'] <= upper)]


# In[29]:


#checking if we still have the outliers using boxplot
sns.boxplot(data=df1, x='Mar Cap - Crore')


# In[37]:


#I will use IQR method to eliminate the outliers
#calculating first and third quartiles
Q1 = df1['Mar Cap - Crore'].quantile(0.25)
Q3 = df1['Mar Cap - Crore'].quantile(0.75)

#Calculating IQR
IQR = Q3-Q1

#defining lower and upper limits
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

#filtering the dataset to remove the outliers
df1 = df1[(df1['Mar Cap - Crore'] >= lower) & (df1['Mar Cap - Crore'] <= upper)]


# In[38]:


#checking if we still have the outliers using boxplot
sns.boxplot(data=df1, x='Mar Cap - Crore')


# In[43]:


#finally we removed the outliers from the 'Mar Cap - Crore' column


# In[44]:


#Now I will use IQR method to eliminate the outliers in 'Sales Qtr - Crore' column
#calculating first and third quartiles
Q1 = df1['Sales Qtr - Crore'].quantile(0.25)
Q3 = df1['Sales Qtr - Crore'].quantile(0.75)

#Calculating IQR
IQR = Q3-Q1

#defining lower and upper limits
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

#filtering the dataset to remove the outliers
df1 = df1[(df1['Sales Qtr - Crore'] >= lower) & (df1['Sales Qtr - Crore'] <= upper)]


# In[45]:


#checking if we still have the outliers using boxplot
sns.boxplot(data=df1, x='Sales Qtr - Crore')


# In[46]:


#Now I will repeat IQR method to eliminate the outliers in 'Sales Qtr - Crore' column
#calculating first and third quartiles
Q1 = df1['Sales Qtr - Crore'].quantile(0.25)
Q3 = df1['Sales Qtr - Crore'].quantile(0.75)

#Calculating IQR
IQR = Q3-Q1

#defining lower and upper limits
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

#filtering the dataset to remove the outliers
df1 = df1[(df1['Sales Qtr - Crore'] >= lower) & (df1['Sales Qtr - Crore'] <= upper)]


# In[47]:


#checking if we still have the outliers using boxplot
sns.boxplot(data=df1, x='Sales Qtr - Crore')


# In[48]:


#all the outliers from the column 'Sales Qtr - Crore' has been removed


# In[49]:


#checking the dataset description
df1.describe()


# In[62]:


#I can see that there is some zero values in 'Sales Qtr - Crore' column. Lets replace it with NaN and impute it with KNN imputer
df1['Sales Qtr - Crore'].replace(0, np.nan, inplace=True)


# In[64]:


#checking how many null values are there
df1['Sales Qtr - Crore'].isna().sum()


# In[66]:


#applying KNN imputation method on Sales Qtr - Crore
data = df1[['Sales Qtr - Crore']]
imputer = KNNImputer(n_neighbors=5)
imputed_data = imputer.fit_transform(data)
df1['Sales Qtr - Crore'] = imputed_data


# In[67]:


df1.isna().sum()


# In[68]:


#we can see that there is no numm values left


# In[70]:


#lets check for the outliers again for the column 'Sales Qtr - Crore'


# In[69]:


sns.boxplot(data=df1, x='Sales Qtr - Crore')


# In[71]:


# we can see there is a outlier. So lets remove it using IQR method
#calculating first and third quartiles
Q1 = df1['Sales Qtr - Crore'].quantile(0.25)
Q3 = df1['Sales Qtr - Crore'].quantile(0.75)

#Calculating IQR
IQR = Q3-Q1

#defining lower and upper limits
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

#filtering the dataset to remove the outliers
df1 = df1[(df1['Sales Qtr - Crore'] >= lower) & (df1['Sales Qtr - Crore'] <= upper)]


# In[72]:


sns.boxplot(data=df1, x='Sales Qtr - Crore')


# In[73]:


#now there is no outliers left


# In[74]:


#lets use describe one last time
df1.describe()


# In[75]:


#now the data looks good
#lets export it to excel file


# In[76]:


df1.to_csv('C:\\Users\\krishna kant\\Downloads\\financial_data_analysis.csv', index=False)


# In[ ]:




