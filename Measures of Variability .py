#Measures of Variability A measure of variability is a summary statistic that represents the amount of dispersion in a dataset.

#Range , Interquartile Range,  Variance , Standard Deviation

# use sepal length

import pandas as pd
from statistics import stdev,variance
df=pd.read_csv('Iris.csv')
X=df['SepalLengthCm']
R=max(X)-min(X)
m=X.mean()
s=stdev(X)
IQR=X.quantile(0.75)-X.quantile(0.25)
o=variance(X)
print(R)
print(m)
print(s)
print(IQR)
print(o)
