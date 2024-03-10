import pandas as pd
import numpy as np
import streamlit as st
import pickle


st.title('Sentiments analysis')
model=pickle.load(open(r"C:\Users\kittu\model_nlp.pk1","rb"))
review_bow=pickle.load(open(r"C:\Users\kittu\review_bow","rb"))
review=st.text_input("enter a review")
data88=pd.DataFrame({"review1":[review]})
review_=review_bow.transform(data88["review1"])


result=model.predict(review_)
if st.button('predict'):
    if result>3:
        st.write("postive sentiment")
    else:
        st.write("negative sentiment")
