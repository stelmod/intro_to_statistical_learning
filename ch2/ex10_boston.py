
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:

boston = load_boston()


# In[3]:

print(boston.DESCR)


# In[4]:

data = pd.DataFrame(data=boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target


# In[5]:

data.head(5)


# In[6]:

data.shape


# In[7]:

data.describe()


# In[8]:

sns.pairplot(data, x_vars=["CRIM", "PTRATIO", "INDUS", "AGE"], y_vars=data.keys())


# In[9]:

more_than7_rooms = data[data['RM'] > 7]
more_than8_rooms = data[data['RM'] > 8]


# In[10]:

more_than7_rooms.describe()


# In[11]:

more_than8_rooms.describe()


# In[12]:

data.ix[data['AGE'].idxmin()]

