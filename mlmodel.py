
'''
This is a simple linear regression model to predit the CO2 emmission from cars
Dataset:
FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions
for new light-duty vehicles for retail sale in Canada
'''

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv('ia_dataset.csv')


df=df.drop(columns={'USN','NAME'})
df=df.rename(columns={'FINAL_EXAM(60)':'FINAL_EXAM'})
df.head()

x=df.IA1
y='df.FINAL_EXAM(60)'
plt.scatter(df.IA1,df.FINAL_EXAM)
plt.scatter(df.IA2,df.FINAL_EXAM)
plt.scatter(df.IA3,df.FINAL_EXAM)

reg=linear_model.LinearRegression()


#Fitting model with trainig data
reg.fit(df.drop('FINAL_EXAM',axis='columns'),df.FINAL_EXAM)

# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(reg, open('model.pkl','wb'))

