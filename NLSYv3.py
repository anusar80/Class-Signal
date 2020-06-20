#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 06:07:04 2020

@author: anusarfarooqui
"""

import os
    
os.chdir('/Users/anusarfarooqui/Docs/Matlab/')
import pandas as pd
#import matplotlib.pyplot as plt


df = pd.read_excel('NLSYv1.xlsx',header=0)

#import xlsxwriter as write
#import scipy.stats as stats
from numpy import mean
from numpy import var
from math import sqrt
#from numpy import std
import statsmodels.api as sm
import numpy as np
#import matplotlib.pyplot as plt
#from numpy import log¬
# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
	return (u1 - u2) / s

#%% Class bias in risk factors: education as class signifier
df['LowerClass'] = df.GRADE < round(np.percentile(df.GRADE.dropna(),25))
df['WorkingClass'] = np.logical_and(df.GRADE > round(np.percentile(df.GRADE.dropna(),25)) - 1,
  df.GRADE < round(np.percentile(df.GRADE.dropna(),75)))
df['MiddleClass'] = df.GRADE > round(np.percentile(df.GRADE.dropna(),75)) - 1

dfClean = pd.DataFrame({'EverArrested':df.EVERARRESTED.values})
# explananda 
dfClean['Arrest'] = df.EVERARRESTED
dfClean['Deal'] = df.EverSellWeed
dfClean['Smoke'] = df.EverSmoked
dfClean['MarijuanaUse'] = df.EverUsedMarijuana
dfClean['DrugUse'] = df.EverUsedDrugs
dfClean['Gang'] = df.GangMember
# features
dfClean['Female'] = df.SEX=='Female'
#dfClean['RACE'] = df.RACE
dfClean['Black'] = df.BLACK
dfClean['Hispanic'] = df.HISP
#dfClean['Mixed'] = df.RACE=='Mixed'
#dfClean['White'] = df.RACE=='White'
dfClean['Grade'] = df.GRADE
dfClean['IQ'] = df.ASVAB
#dfClean['GPA'] = df.GPA
dfClean['NoDiploma'] = df.LowerClass
dfClean['HighSchool'] = df.WorkingClass
dfClean['College'] = df.MiddleClass


# statsmodels is weak on missing data
dfClean = dfClean.dropna()


y = y=np.vstack((dfClean.Arrest,dfClean.Deal,dfClean.Smoke,dfClean.MarijuanaUse,dfClean.DrugUse,dfClean.Gang)).T

X = np.vstack((np.ones((1,np.size(dfClean.Grade,0))),dfClean.Female,dfClean.Grade,dfClean.IQ,dfClean.NoDiploma,dfClean.College,dfClean.Black,dfClean.Hispanic)).T
X = X[:,[0,1,2]]

b = np.zeros((np.size(y,1),np.size(X,1),2))

for i in range(np.size(y,1)):
    glm = sm.GLM(y[:,i],X,family=sm.families.Binomial()).fit()
    for j in range(np.size(X,1)):
        b[i,j,0] = glm.params[j]
        b[i,j,1] = glm.bse[j]

writer = pd.ExcelWriter('RiskTaking01.xlsx', engine='xlsxwriter')

for i in range(2):
    BD = pd.DataFrame(b[:,:,i])
    BD.to_excel(writer, sheet_name='a%d' % i)

writer.save()

X = np.vstack((np.ones((1,np.size(dfClean.Grade,0))),dfClean.Female,dfClean.Grade,dfClean.IQ,dfClean.NoDiploma,dfClean.College,dfClean.Black,dfClean.Hispanic)).T
X = X[:,[0,1,2,3]]

b = np.zeros((np.size(y,1),np.size(X,1),2))

for i in range(np.size(y,1)):
    glm = sm.GLM(y[:,i],X,family=sm.families.Binomial()).fit()
    for j in range(np.size(X,1)):
        b[i,j,0] = glm.params[j]
        b[i,j,1] = glm.bse[j]

writer = pd.ExcelWriter('RiskTaking02.xlsx', engine='xlsxwriter')

for i in range(2):
    BD = pd.DataFrame(b[:,:,i])
    BD.to_excel(writer, sheet_name='b%d' % i)

writer.save()

X = np.vstack((np.ones((1,np.size(dfClean.Grade,0))),dfClean.Female,dfClean.Grade,dfClean.IQ,dfClean.NoDiploma,dfClean.College,dfClean.Black,dfClean.Hispanic)).T
X = X[:,[0,1,3,4,5]]

b = np.zeros((np.size(y,1),np.size(X,1),2))

for i in range(np.size(y,1)):
    glm = sm.GLM(y[:,i],X,family=sm.families.Binomial()).fit()
    for j in range(np.size(X,1)):
        b[i,j,0] = glm.params[j]
        b[i,j,1] = glm.bse[j]

writer.save()

writer = pd.ExcelWriter('RiskTaking03.xlsx', engine='xlsxwriter')

for i in range(2):
    BD = pd.DataFrame(b[:,:,i])
    BD.to_excel(writer, sheet_name='c%d' % i)
    
X = np.vstack((np.ones((1,np.size(dfClean.Grade,0))),dfClean.Female,dfClean.Grade,dfClean.IQ,dfClean.NoDiploma,dfClean.College,dfClean.Black,dfClean.Hispanic)).T
X = X[:,[0,1,3,4,5,6,7]]

b = np.zeros((np.size(y,1),np.size(X,1),2))

for i in range(np.size(y,1)):
    glm = sm.GLM(y[:,i],X,family=sm.families.Binomial()).fit()
    for j in range(np.size(X,1)):
        b[i,j,0] = glm.params[j]
        b[i,j,1] = glm.bse[j]
        
writer.save()

writer = pd.ExcelWriter('RiskTaking04.xlsx', engine='xlsxwriter')

for i in range(2):
    BD = pd.DataFrame(b[:,:,i])
    BD.to_excel(writer, sheet_name='d%d' % i)
    
writer.save()

X = np.vstack((np.ones((1,np.size(dfClean.Grade,0))),dfClean.Female,dfClean.Grade,dfClean.IQ,dfClean.NoDiploma,dfClean.College,dfClean.Black,dfClean.Hispanic)).T
X = X[:,[0,1,2,3,6,7]]

b = np.zeros((np.size(y,1),np.size(X,1),2))

for i in range(np.size(y,1)):
    glm = sm.GLM(y[:,i],X,family=sm.families.Binomial()).fit()
    for j in range(np.size(X,1)):
        b[i,j,0] = glm.params[j]
        b[i,j,1] = glm.bse[j]

writer = pd.ExcelWriter('RiskTaking05.xlsx', engine='xlsxwriter')

for i in range(2):
    BD = pd.DataFrame(b[:,:,i])
    BD.to_excel(writer, sheet_name='e%d' % i)

writer.save()