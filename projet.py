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

# Regression linéaire multiple sur données SIDA:

df_aids = pd.read_csv("C:/Users/lenovo/Documents/aidscsv")
df_aids  # chargement des données 
aids_mixed_all = smf.mixedlm("CD4 ~ obstime+drug+gender+prevOI+AZT", df_aids, 
                             groups=df_aids["id"]) 
mdf = aids_mixed_all.fit(method=["lbfgs"])
print(mdf.summary())  # regression lineaire multiple

# selection des variables pour la regression linéaire simple
x = df_aids.drop(columns=["Unnamed: 0", "id", "time", "death", "CD4"])  
y = df_aids["CD4"]
# création d'échatillon d'entraienemnt et de test
x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size = 0.2, 
                                                   random_state = 10)

aids_lm_all = linear_model.LinearRegression()
aids_lm_all.fit(x_train, y_train)
print(aids_lm_all.coef_)  # coefficient éstimés par  regression linéaire
print(aids_lm_all.intercept_)  # intercetpe éstimé par regression linéaire
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
# création d'echantillon d'apprentissage et de test
x_train, x_test, y_train ,y_test = train_test_split(x, y,  test_size = 0.2, 
                                                  random_state = 10)

lm = linear_model.LinearRegression()  
lm.fit(x_train, y_train)
print(lm.coef_)  # coefficients etimés par la regression linéaire
print(lm.intercept_)  # valeur de l'intercepte