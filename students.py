# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('student-mat.csv', sep=';')

plt.figure(figsize=(11,11))
sns.heatmap(df.corr().round(1), annot=True)

df = df[['failures','G1','G2','G3']]
print(df)
X = df.iloc[:, 0:3].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))


y_pred = reg.predict(X_test)
plt.scatter(y_test, y_pred)

import numpy as np
pred = np.array([3, 14, 17]).reshape(1,-1)
print(reg.predict(pred))