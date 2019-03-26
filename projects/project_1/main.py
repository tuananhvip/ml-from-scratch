import pandas as pd
import numpy as np
import seaborn as sb

df = pd.read_csv('data/agaricus-lepiota.csv')

X = df.drop('class', axis=1).values

y = df.loc[:, 'class'].values

columns = df.drop('class', axis=1).columns

stalk_root = df.loc[:, 'stalk-root'].values

print("Percentage missing value in stalk-root column", len(stalk_root[stalk_root == '?'])/len(stalk_root))

sb.pairplot(X)