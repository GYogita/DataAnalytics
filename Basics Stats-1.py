#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Basic Stats No 1 Assignment


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv('sales_data_with_discounts.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# ## Descriptive Analytics for Numerical Columns

# In[6]:


df.info()


# ------------------------------------------------------------------------------------------------------------------
# 1. Numerical columns: Volume, Avg Price, Total Sales Value, Discount Rate (%), Discount Amount , Net Sales Value 
# 2. categorical columns: Date, Day, SKU, City, BU, Brand, Model

# In[8]:


# Mean of all numeric columns
means = df.mean(numeric_only=True)

# Median of all numeric columns
medians = df.median(numeric_only=True)

# Mode of all columns
modes = df.mode(numeric_only=False).iloc[0]  # Take the first mode if multiple

# Standard Deviation
std=df.std(numeric_only=True)

# Display the results
print("Means:\n", means)
print("\nMedians:\n", medians)
print("\nModes:\n", modes)
print("\nStandard Deviation:\n", std)
    


# ## Data Visualization

# In[20]:


#Distribution for numeric columns
for col_name in df.select_dtypes('number'):
    sns.histplot(df[col_name], kde=True)
    plt.title(f'Distribution of {col_name}')
    plt.show()
    


# In[10]:


# Box plots
for col_name in df.select_dtypes('number'):
    print(col_name)
    sns.boxplot(df[col_name])
    plt.show()


# ### Analysis:
# 1. Volume : Right skewed with positive outliers
# 2. Avg Price: Right skewed with some positive outliers
# 3. Sales Value: Right skewed with positive outliers
# 4. Discount Rate: Left skewed with negative outliers
# 5. Discount Amount: Right skewed with positive outliers
# 6. Net Sales value: Right skewed with positive outliers

# In[16]:


for col_name in df[['Brand','BU']]:
    print(col_name)
    sns.countplot(df[col_name])
    plt.show()


# ### Count plot analysis:
# Brand Jeera is having highest entries and Samsung, orange stand at lowest as compared to other brands
# Mobile, FMCG, Life style are having similar counts 

# In[21]:


Brand_wise_sales=np.round(df.groupby('Brand')['Total Sales Value'].sum(),2) # select region,sum(profit) from df group by region
plt.title('Brand wise Total Sales Value')
plt.bar(Brand_wise_sales.index, Brand_wise_sales.values)
plt.xlabel('Brand')
plt.ylabel('Total Sales Value')


# In[24]:


BU_wise_sales=np.round(df.groupby('BU')['Total Sales Value'].sum(),2) # select region,sum(profit) from df group by region
plt.title('BU wise Total Sales Value')
plt.pie(BU_wise_sales.values, labels= BU_wise_sales.index, autopct="%0.2f%%")


# In[23]:


Day_wise_sales=np.round(df.groupby('Day')['Total Sales Value'].sum(),2) 
plt.title('BU wise Total Sales Value')
plt.bar(Day_wise_sales.index, Day_wise_sales.values)
plt.xlabel('Day')
plt.ylabel('Total Sales Value')


# In[27]:


BU_wise_discounts=np.round(df.groupby('BU')['Discount Amount'].sum(),2)
plt.title('BU wise Discount')
plt.pie(BU_wise_discounts.values, labels= BU_wise_discounts.index, autopct="%0.2f%%")
plt.show()


# # Standardization of Numerical Variables

# Standardization, also known as z-score normalization, is a data preprocessing technique used to transform the features of a dataset so they have the properties of a standard normal distribution — specifically:
# 
# A mean (μ) of 0
# 
# A standard deviation (σ) of 1

# In[29]:


from sklearn.preprocessing import StandardScaler


# In[30]:


num_col= df.select_dtypes('number').columns


# In[31]:


num_col


# In[33]:


scalar=StandardScaler()
df[num_col]=scalar.fit_transform(df[num_col])


# In[41]:


df[num_col].head()


# In[43]:


for col_name in df[num_col]:
    print(col_name,round(np.mean(df[num_col])))


# Before Means:
#  Volume                   5.066667
# Avg Price            10453.433333
# Total Sales Value    33812.835556
# Discount Rate (%)       15.155242
# Discount Amount       3346.499424
# Net Sales Value      30466.336131

# # Conversion of Categorical Data into Dummy Variables

# Converting categorical data into dummy variables, also known as one-hot encoding, is essential because most machine learning algorithms and statistical models require numerical input.
# One-hot encoding transforms a categorical feature with n unique values into n binary (0 or 1) columns. Each column represents one category, and a 1 is placed in the column corresponding to the category for that row, while the rest are 0.

# In[44]:


df.head()


# In[51]:


#deleting irrelevant columns
df=df.drop(['Date','SKU','City','Model'], axis=1, inplace=True)


# In[52]:


df.head()


# In[53]:


df=pd.get_dummies(df, dtype=int, drop_first=True) # each textual data conversion to individual column
#  drop_first=True removes duplicate information Eg. only keeps Sex_Male or Sex_female


# In[54]:


df.head()


# ### Now this data frame can be used to build the machine learning model

# One-hot encoding transforms a categorical feature with n unique values into n binary (0 or 1) columns. Each column represents one category, and a 1 is placed in the column corresponding to the category for that row, while the rest are 0.
# so here it has transformed Day, BU, Brand columns in unique values.

# In[ ]:




