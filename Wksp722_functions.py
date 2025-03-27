# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 09:33:56 2022

@author: hpatel
"""
def M3L2_1():
    print('# Replace for loop') 
    print('df3 = df.copy()')
    print('ind = df3.loc[df3.loc[:,\'Age\'].isnull(),:].index')
    print('df3.loc[ind,\'Age\'] = df3.loc[ind,\'Salutation\'].map(median_age)')
    print('#check if both dataframes are equal:')
    print('df.equals(df3)')
    
def M3L2_2():
    print('Survivors = df.loc[df.loc[:,\'Survived\']==1,:]')
    print('Grouped_Survivors = Survivors.groupby(\'Sex\')')
    print('Grouped_Survivors_Count = Grouped_Survivors.count().loc[:,\'Name\']')
    print('totalPassengers = df.shape[0]')
    print('Pcnt_Grouped_Survivors_Count = 100 * Grouped_Survivors_Count / totalPassengers') 
    print('Pcnt_Grouped_Survivors_Count')
    
    print('\n# or\n') 
    print('df.loc[df.loc[:,\'Survived\']==1,:].groupby(\'Sex\').count().loc[:,\'Name\'] / df.shape[0] * 100')
    
def M3L2_3():
    print('Survivors = df.loc[df.loc[:,\'Survived\']==1,:]')
    print('Grouped_Survivors = Survivors.groupby(\'Pclass\')')
    print('Grouped_Survivors_Count = Grouped_Survivors.count().loc[:,\'Name\']')
    print('totalPassengers = df.shape[0]')
    print('Pcnt_Grouped_Survivors_Count = 100 * Grouped_Survivors_Count / totalPassengers') 
    print('Pcnt_Grouped_Survivors_Count')
    
    print('\n# or\n') 
    print('df.loc[df.loc[:,\'Survived\']==1,:].groupby(\'Pclass\').count().loc[:,\'Name\'] / df.shape[0] * 100')
    
#*****************************************************************************
# Functions for use in the notebooks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


# Function below is from M3L3, and summarizes changes to the dataframe to turn categorical variables into numerical ones
def titanicNumericalConverter(df):
    # convert the categorical variable 'Sex' to numerical 0 and 1 using mapping
    mapping = {'male':0, 'female':1}
    df.loc[:,'Sex'] = df.Sex.map(mapping)
    
    #convert columns using one-hot state encoding:
    dfTemp = pd.get_dummies(df.loc[:,['Embarked','Salutation']])
    df = pd.concat([df,dfTemp], axis=1)
    df.drop(['PassengerId','Embarked','Name','Ticket','Salutation'], axis=1,inplace=True)
    return df

# Function below is from M3L3, and summarizes changes to the dataframe to turn categorical variables into numerical ones
def titanicNumericalConverter2(df):
    # convert the categorical variable 'Sex' to numerical 0 and 1 using mapping
    mapping = {'male':0, 'female':1}
    df.loc[:,'Sex'] = df.Sex.map(mapping)
    
    #convert columns using one-hot state encoding:
    dfTemp = pd.get_dummies(df.loc[:,['Embarked','Salutation']])
    df = pd.concat([df,dfTemp], axis=1)
    df.drop(['Embarked','Name','Ticket','Salutation'], axis=1,inplace=True)
    return df

# Function below plots the test error and optional oob error as the forest grows
def EnsembleGrowthErrorPlot(clf,x_train,y_train,x_test,y_test,min_estimators=5,max_estimators=200,oob=False):

    # plot the oob score as you add trees
    oob_error_rate = []
    test_error_rate = []
    for i in range(min_estimators, max_estimators + 1, 5):
            clf.set_params(n_estimators=i)
            clf.fit(x_train, y_train)
            if oob==True:
                # Record the OOB error for each `n_estimators=i` setting.
                oob_error = 1 - clf.oob_score_
                oob_error_rate.append((i, oob_error))

            # Record the test set error rate
            y_pred = clf.predict(x_test)
            test_error = 1 - metrics.accuracy_score(y_test, y_pred)
            test_error_rate.append((i,test_error))

    # Plot the Out of Bag Error and the Test set Error rate as a function of # of trees
    fig, axes = plt.subplots()
    if oob==True:
        x,oob_error = zip(*oob_error_rate)
        axes.plot(x, oob_error, label='oob error')
    x,test_error = zip(*test_error_rate)
    axes.plot(x, test_error, label='test error')
    axes.legend()
    plt.xlabel("Number of Trees")
    plt.ylabel("Error percentage")
    plt.show()

# Insert columns in the test set of Titanic characters.  This function is only here to reduce typing during class
# def M3L3_titanicTest_colInsert(x_test):
#     x_test.insert(loc = 7,column = 'Embarked_Q',value = 0)
#     x_test.insert(loc = 9,column = 'Salutation_Capt.',value = 0)
#     x_test.insert(loc = 10,column = 'Salutation_Col.',value = 0)
#     x_test.insert(loc = 11,column = 'Salutation_Countess.',value = 0)
#     x_test.insert(loc = 12,column = 'Salutation_Don.',value = 0)
#     x_test.insert(loc = 13,column = 'Salutation_Dr.',value = 0)
#     x_test.insert(loc = 14,column = 'Salutation_Jonkheer.',value = 0)
#     x_test.insert(loc = 15,column = 'Salutation_Lady.',value = 0)
#     x_test.insert(loc = 16,column = 'Salutation_Major.',value = 0)
#     x_test.insert(loc = 17,column = 'Salutation_Master.',value = 0)
#     x_test.insert(loc = 19,column = 'Salutation_Mlle.',value = 0)
#     x_test.insert(loc = 20,column = 'Salutation_Mme.',value = 0)
#     x_test.insert(loc = 23,column = 'Salutation_Ms.',value = 0)
#     x_test.insert(loc = 24,column = 'Salutation_Rev.',value = 0)
#     x_test.insert(loc = 25,column = 'Salutation_Sir.',value = 0)
#     return x_test

def M3L3_titanicTest_colInsert(x_test):
    temp = pd.read_csv('titanic_train_columns.csv')
    temp = temp.iloc[:,2:]
    cols = temp.columns.tolist()

    ind=0
    for i in cols:
        if i not in x_test:
            print(i,ind)
            x_test.insert(loc = ind,column = i,value = 0)
        ind = ind+1
    return x_test

# Scatter plot of California housing values and population
# derived from https://www.kaggle.com/code/mostafaashraf1/california-housing-prices
def M3L4_CA_plot(df_housing):
    # Scatter plot between longitude and latitude
    plt.figure(figsize=(12,6))
    sc = plt.scatter(df_housing["longitude"],
                     df_housing["latitude"],
                     alpha=0.4,
                     cmap="viridis",
                     c=df_housing["median_house_value"],
                    s=df_housing["population"]/50,
                    label='population')
    plt.colorbar(sc)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.title("longitude vs latitude")
    plt.legend()
    plt.show()
    
def M3L4_Predicted_Plot(x_test, y_test, y_pred, numPts = 100):
    indices = np.argsort(y_pred)
    y_pred = y_pred[indices]
    y_test = y_test.iloc[indices]
    
    # Let's see how well we did
    fig, ax = plt.subplots()

    
    ax.plot(np.arange(0,numPts),y_test[0:numPts] ,color='b', label='Original')
    ax.plot(np.arange(0,numPts),y_pred[0:numPts], linewidth=2,color='r',label='Predicted')
    ax.legend()
    plt.show()

