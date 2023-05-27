import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st

df=pd.read_csv(r"C:\Users\ajuaj\Downloads\mul\Language Detection.csv")

text=df.Text

def pun(text):
    for pun in string.punctuation :
       text = text.replace(pun,'')
    text=text.lower()
    return(text)
text.apply(pun)

X=df.iloc[:,0]
y=df.iloc[:,1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

vec=TfidfVectorizer(ngram_range=(1,2),analyzer='char')
clf=LogisticRegression()

model_pipe=pipeline.Pipeline([('vec',vec),('clf',clf)])

model_pipe.fit(X_train,y_train)
pred=model_pipe.predict(X_test)

accuracy_score(y_test,pred) 

f=open('model.pkl','wb')
pickle.dump(model_pipe,f)
f.close()

f=open(r'C:\ComputerVision\Lang_detect\model.pkl','rb')
fm=pickle.load(f)
f.close

st.title("Langauage Detection Model")
text=st.text_input("Enter your Text")

button_click=st.button("Press Here")
if button_click :
    st.text(fm.predict([text]))