# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:43:00 2020

@author: lenovo
"""

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold
import statsmodels.formula.api as smf
import requests
from sklearn.utils import Bunch
from sklearn import datasets 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA

# Regression linéaire multiple sur données aids:

df_aids = pd.read_csv("C:/Users/lenovo/Documents/aidscsv")
df_aids  # chargement des données 
aids_mixed_all = smf.mixedlm("CD4 ~ obstime+drug+gender+prevOI+AZT", df_aids, 
                             groups=df_aids["id"]) 
mdf = aids_mixed_all.fit(method=["lbfgs"])
print(mdf.summary())  # regression lineaire multiple

# Regression linéaire sur les données aids:
# Recodage des variables qualitatives:
d1 = pd.get_dummies(df_aids["drug"])
d2 = pd.get_dummies(df_aids["gender"])
d3 =pd.get_dummies(df_aids["prevOI"])
d4 = pd.get_dummies(df_aids["AZT"])
# Concaténation successives des nouvelles varibles
df = pd.concat([df_aids,d1],axis=1)
df1 = pd.concat([df,d2],axis=1)
df2 = pd.concat([df1,d3],axis=1)
df3 = pd.concat([df2,d4],axis=1)
df_aids = df3
# selection des variables pour la regression linéaire
x = df_aids.drop(columns=["Unnamed: 0", "id", "time", "death", "CD4", "drug",
                          "gender","prevOI","AZT"])  
y = df_aids["CD4"]
# Regression OLS avec statsmodel: 
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
df_aids.boxplot('CD4')  # bocplot de la variable à predire

# Regression linéaire classique sur données Boston:

boston = datasets.load_boston()  
print(boston.keys())
boston.feature_names  # afficher les noms des variables explicatives
df = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])
print(df)  # renommer les colonnes par le noms des variables explicatives
df['MEDV'] = boston['target']  # la variable a expliquer
y = df.MEDV.copy()
del df['MEDV']
df_boston = pd.concat((y, df), axis=1)  # ajout de la variable à predire au tableau
print(df_boston)
# Histogramme de la fréquence des nombres de chambres dans les maisons
plt.figure(figsize=(8, 8))
plt.hist(df_boston['RM'], density=True, bins=50)
plt.xlabel("Nombre de chambres")
plt.ylabel("Fréquence")
plt.title("Fréquence des nombres de chambres")

# Selection des variables prédictives dont on a besoin
x = df_boston.drop(columns=["MEDV", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                            "DIS", "RAD", "PTRATIO", "B"])
y = boston.target

lm = linear_model.LinearRegression()  # regression linéaire
lm.fit(x,y)
print(lm.coef_)  # coefficients etimés par la regression linéaire
print(lm.intercept_)  # valeur de l'intercept

# Test sur les séries chronologiques données ChickEgg:

df_ce = pd.read_csv("C:/Users/lenovo/Documents/chickeggcsv")
df_ce.head()  # chargement des données

# representation de la serie chronologique pour les poulets:
ax = sns.lineplot(x='Unnamed: 0',y='chicken',data=df_ce)

# representation de la serie chronologique des oeufs:
ax1 = sns.lineplot(x='Unnamed: 0',y='egg',data=df_ce)

#test de granger des valeurs des oeufs a partir de celle des poulets:
model1 = grangercausalitytests(df_ce[["chicken","egg"]], maxlag=3)

#test de granger des valeurs des poulets à partir de celle des oeufs:
model2 = grangercausalitytests(df_ce[["egg","chicken"]], maxlag=3)