import numpy as np;
import matplotlib.pyplot as plt
import pandas as pd;
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder;

dataset = pd.read_csv('Data.csv');
X = dataset.iloc[:,:-1].values;
Y = dataset.iloc[:,3].values;
Imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis=0);
Imputer.fit(X[:,1:3]);
X[:,1:3] = Imputer.transform(X[:,1:3]);
labelencoder_X= LabelEncoder();
X[:,0] = labelencoder_X.fit_transform(X[:,0]);
onehotencoder = OneHotEncoder(categorical_features=[0]);
X = onehotencoder.fit_transform(X).toarray();