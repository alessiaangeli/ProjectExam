# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:34:09 2019

@author: aless
"""

"""
Project work of the course DIGITAL EPIDEMIOLOGY

AUTHORS: Emanuela Iovino, Alessia Angeli, Paola Dimartino, Erika Gardini

GROUND TRUTH: depressione (crude prevalence by region in usa)

DIGITAL PROXY DATA: antidepressivo, autostima, barbiturato, creeper, depressione, facebook, fluoxetina, incel, ketamina, paroxetina, prozac, sertralina

YEARS: 2016, 2017, 2018
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns


#Read from csv
data2016 = pd.read_csv("data2016.csv")
data2016 = data2016.fillna(0)
data2017 = pd.read_csv("data2017.csv")
data2017 = data2017.fillna(0)
data2018 = pd.read_csv("data2018.csv")
data2018 = data2018.fillna(0)


print("Correlation matrix gt vs proxy data - 2016")
corr2016 = data2016.corr()
ax = sns.heatmap(corr2016, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20,220,n=200), square=True)
ax.set_xticklabels(data2016.columns[1:],rotation=45, horizontalalignment='right')
plt.show()

print("Correlation matrix gt vs proxy data - 2017")
corr2017 = data2017.corr()
ax = sns.heatmap(corr2017, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20,220,n=200), square=True)
ax.set_xticklabels(data2017.columns[1:],rotation=45, horizontalalignment='right')
plt.show()

print("Correlation matrix gt vs proxy data - 2018")
corr2018 = data2018.corr()
ax = sns.heatmap(corr2018, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20,220,n=200), square=True)
ax.set_xticklabels(data2018.columns[1:],rotation=45, horizontalalignment='right')
plt.show()

print("We can observe that the correlation matrices are similar across different years.")
print("")


def fit_the_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("r_sq ", str(model.score(x_train, y_train)))
    return model

def eval_model(x_test, y_test, model):
    y_pred = model.predict(x_test)

    error = (np.abs(y_test - y_pred) / y_test) * 100
    result = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten(), 'Error(%)': error})
    print(result)
    return error


file_names = ["data2016", "data2017", "data2018"]
global_error = []
mean_error_per_year = []
global_error_income = []
mean_error_per_year_income = []

### Cross validation: we repeat the experiment three times, using as training set the data from two years 
### and as test set the remaining data
for el in file_names:
    
    test = pd.read_csv(el + ".csv")
    test = data2016.fillna(0)

    train_name = list(filter(lambda x: (x != el), file_names))

    train1 = pd.read_csv(train_name[0] + ".csv")
    train1 = train1.fillna(0)
    train2 = pd.read_csv(train_name[1] + ".csv")
    train2 = train2.fillna(0)


    train1 = train1.values
    train2 = train2.values
    train = np.concatenate((train1, train2))
    x_train = train[:, 2:-1]
    x_train_income = train[:, 2:]
    y_train = train[:, 1]

    test = test.values
    x_test = test[:, 2:-1]
    x_test_income = test[:, 2:]
    y_test = test[:, 1]

    print("TRAIN: " + train_name[0] + " " + train_name[1])
    print("TEST: " + el)
    
    print("Compute linear regressor without income...")
    model = fit_the_model(x_train, y_train)
    print("Show error of the model")
    error = eval_model(x_train, y_train, model)
    global_error.append(error)
    mean_error_per_year.append(np.mean(error))
    
    print("Compute linear regressor with income...")
    model = fit_the_model(x_train_income, y_train)
    print("Show error of the model")
    error = eval_model(x_train_income, y_train, model)
    global_error_income.append(error)
    mean_error_per_year_income.append(np.mean(error))
    
    
global_mean_error = np.array(global_error)
global_mean_error = np.mean(global_mean_error)
mean_error_per_year = np.array(mean_error_per_year)
print("Global mean error without income " + str(global_mean_error))
print("Mean error per year without income " + str(mean_error_per_year))

print("\n")

global_mean_error_income = np.array(global_error_income)
global_mean_error_income = np.mean(global_mean_error_income)
mean_error_per_year_income = np.array(mean_error_per_year_income)
print("Global mean error with income " + str(global_mean_error_income))
print("Mean error per year with income " + str(mean_error_per_year_income))

print("")
print("We can observe that adding the information about the householding income the performance of the model is not improved.")
print("")

if(global_mean_error!=0 and len(mean_error_per_year)!=0 and global_mean_error_income!=0 and len(mean_error_per_year_income)!=0):
    print("Script correctly executed")