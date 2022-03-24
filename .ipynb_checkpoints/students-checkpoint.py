# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('student-mat.csv', sep=';')

plt.figure(figsize=(11,11))
sns.heatmap(df.corr().round(1), annot=True)

df = df[['failures','G1','G2','G3']]