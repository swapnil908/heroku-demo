import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset=pd.read_csv("hiring.csv")
dataset["experience"].fillna(0,inplace=True)
dataset["test_score(out of 10)"].fillna(dataset["test_score(out of 10)"].mean(),inplace=True)
dataset["salary($)"].fillna(0, inplace =True)
X=dataset.iloc[:,:3]
def convert_to_int(word):
    word_dict={'one':1,"two":2,"three":3,'five':5,'seven':7,'ten':10,"eleven":11,0:0}
    return word_dict[word]
X['experience']=X['experience'].apply(lambda x:convert_to_int(x))
y=dataset.iloc[:,-1]
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)

pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))