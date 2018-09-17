# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import pandas as pd
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import seaborn as sns




#3.0 GET DATA
data = pd.read_csv("weatherHistory.csv")






#Verificar existencia de dados nulos
print(data.isnull().sum())

#Renomear colunas
data.columns = data.columns.str.replace(r"\(.*\)","")
data.columns = data.columns.str.rstrip()
data.columns = data.columns.str.lower()
data.columns = data.columns.str.replace(" ", "_")
data.info()

#Verificar colunas NA
print(data.isnull().sum())

#Verificar os dados de Formated Date
print(data.formatted_date.value_counts())

#Remover duplicatas em Formated Date
data = data.drop_duplicates("formatted_date")
print(data.formatted_date.value_counts())

#Verificar coluna precip_type (517 linhas NA)
data.precip_type.value_counts()
data.info()
sum(data.precip_type.isnull())
data = data.dropna(subset=['precip_type'])
data.info()

#Verificar coluna Temperatura
data.temperature.value_counts()
sum(data.temperature.isnull())

#Verificar coluna loud_cover
print(data.loud_cover.value_counts())
#Remover colunas em branco
data = data.drop("loud_cover", axis=1)
data.info()

#Verificar coluna daily_summary
data.daily_summary.value_counts()

#Descricao dos dados
data.describe()



#3.2.1 HISTOGRAM
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(8,6))
plt.tight_layout()
plt.show()



#Remover colunas categoricas para test (formatted_date, summary, daily_summary)
data = data.drop("formatted_date", axis=1)
data = data.drop("summary", axis=1)
data = data.drop("daily_summary", axis=1)

data.info()

#3.2.2 CREATE A TEST SET

def split_train_test(data, test_ratio):
  
  #scramble the position
  shuffled_indices = np.random.permutation(len(data))
  #find the test set size
  test_set_size = int(len(data) * test_ratio)
  #split the indices
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(data, 0.2)
print("data has {} atributes, that is the same of {}\
train instances + {} test intances ({})".
      format(len(data),len(train_set),len(test_set),len(train_set)+len(test_set)))

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, 
                                       test_size=0.2, 
                                       random_state=35)

print("data has {} instances\n {} train instances\n {} test intances".
      format(len(data),len(train_set),len(test_set)))




#4.0 DISCOVER AND VISUALIZE THE DATA TO GAIN INSIGHTS

train = train_set.copy()





#4.2 LOOKING FOR CORRELATIONS

#Pergunta 1 - Is there a relationship between humidity and temperature? 
corr_matrix = train.corr()
corr_matrix["temperature"].sort_values(ascending=False)

#Pergunta 2 - What about between humidity and apparent temperature? 
corr_matrix["humidity"].sort_values(ascending=False)

#Pergunta 3 - Can you predict the apparent temperature given the humidity?
corr_matrix["apparent_temperature"].sort_values(ascending=False)

sns.heatmap(train.corr(), annot=True, fmt=".2f")

columns = ["temperature", "humidity", "apparent_temperature"]
sns.pairplot(train[columns], diag_kind='hist')






#5.0 Prepare the Data for Machine Learning Algorithms

# just to remind ...
# train_set, test_set = train_test_split(data, test_size=0.2, random_state=35)

# drop creates a copy of the remain data and does not affect train_set
train_X = train_set.drop("apparent_temperature", axis=1)

# copy the label (y) from train_set
train_y = train_set.humidity.copy()







#5.1 Data Cleaning

# count the number of missing values
train_X.isnull().sum()

# First, you need to create an Imputer instance, specifying that you want 
# to replace each attribute’s missing values with the median of that attribute:
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")

# Since the median can only be computed on numerical attributes, we need to 
# create a copy of the data without the text attribute ocean_proximity:
train_X_num = train_X.drop("precip_type", axis=1)

# Now you can fit the imputer instance to the training data using 
# the fit() method:
imputer.fit(train_X_num)

imputer.statistics_
train_X_num.median().values
# Now you can use this “trained” imputer to transform the training set by 
# replacing missing values by the learned medians:
train_X_num_array = imputer.transform(train_X_num)

# The result is a plain Numpy array containing the transformed features. 
# If you want to put it back into a Pandas DataFrame, it’s simple:
train_X_num_df = pd.DataFrame(train_X_num_array, columns=train_X_num.columns)

train_X_num_df.isnull().sum()


#5.2 Handling Text and Categorical Attributes (formatted_date 95912, summary 27, precip_type 2, daily_summary 214)

#print(data.formatted_date.value_counts())
#print(data.summary.value_counts())
#print(data.precip_type.value_counts())
#print(data.daily_summary.unique().size)


# For this, we can use Pandas' factorize() method which maps each 
# category to a different integer:

train_X_cat_encoded, train_X_categories = train_X.precip_type.factorize()


# train_X_cat_encoded is now purely numerical
train_X_cat_encoded[0:10]

# factorize() method also return the list of categories
train_X_categories

# Scikit-Learn provides a OneHotEncoder encoder to convert 
# integer categorical values into one-hot vectors.

from sklearn.preprocessing import OneHotEncoder 

encoder = OneHotEncoder()

# Numpy's reshape() allows one dimension to be -1, which means "unspecified":
# the value is inferred from the lenght of the array and the remaining
# dimensions
train_X_cat_1hot = encoder.fit_transform(train_X_cat_encoded.reshape(-1,1))

# it is a column vector
train_X_cat_1hot


import sys

print("Using a sparse matrix: {} bytes".format(sys.getsizeof(train_X_cat_1hot.toarray())))
print("Using a dense numpy array: {} bytes".format(sys.getsizeof(train_X_cat_1hot)))




#5.6 Transformation Pipelines

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


num_pipeline = Pipeline([('imputer', Imputer(strategy="median")), ('attribs_adder', CombinedAttributesAdder()), ('std_scaler', StandardScaler())])
train_X_num_pipeline = num_pipeline.fit_transform(train_X_num)
train_X_num_pipeline
