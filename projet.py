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

# Création des groupes
boston_nbig = df_boston[df_boston['RM'] <= 6]
boston_big = df_boston[df_boston['RM'] > 6]

# Selection des variables prédictives dont on a besoin pour big
x = boston_big.drop(columns = ["MEDV","ZN","INDUS","CHAS","NOX","AGE","DIS",
                               "RAD","PTRATIO","B"])  # va explicatives
y = boston_big['MEDV']  # variable à expliquer

# Création d'échantillon d'apprentissage et de test de façon alétoire
x_train,x_test, y_train,y_test = train_test_split( x, y,  test_size = 0.2, 
                                                  random_state = 10)
lm = linear_model.LinearRegression()
lm.fit(x_train,y_train)  # application de la régression linéaire sur apprentissage
print(lm.coef_)  # valeurs estimées des coefficients 
print(lm.intercept_)  # valeur estimée de l'intercept

# On refait la même chose pour 1-big

x = boston_nbig.drop(columns = ["MEDV","ZN","INDUS","CHAS","NOX","RM","AGE",
                                "DIS","RAD","PTRATIO","B"])
y = boston_nbig['MEDV']
x_train,x_test, y_train,y_test = train_test_split( x, y,  test_size = 0.2, 
                                                  random_state = 10)
lm = linear_model.LinearRegression()
lm.fit(x_train,y_train)
print(lm.coef_)
print(lm.intercept_)

# Test sur les séries chronologiques données ChickEgg:

df_ce = pd.read_csv("C:/Users/lenovo/Documents/chickeggcsv")
df_ce.head()  # chargement des données
df_ce = df_ce.rename(columns={'Unnamed: 0': 'Years'})
df_ce.head()

# representation de la serie chronologique pour les poulets:
ax = sns.lineplot(x='Years',y='chicken',data=df_ce)

# representation de la serie chronologique des oeufs:
ax1 = sns.lineplot(x='Years',y='egg',data=df_ce)

#test de granger des valeurs des oeufs a partir de celle des poulets:
model1 = grangercausalitytests(df_ce[["chicken","egg"]], maxlag=3)

#test de granger des valeurs des poulets à partir de celle des oeufs:
model2 = grangercausalitytests(df_ce[["egg","chicken"]], maxlag=3)