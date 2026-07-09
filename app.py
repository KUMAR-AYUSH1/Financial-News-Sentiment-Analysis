import streamlit as st
import requests
import pandas as pd

st.title('Financial News Sentiment Analysis')
url = "http://localhost:8000/predict"


def predict(text):
    response = requests.post(url, json={"text": text}).json()
    return response


def main():
    text = st.text_area("Enter text:", "", height=100)
    if st.button("Predict"):
        response = predict(text)
        st.write("Sentiment is",response['prediction'],"model confidence is ",response['confidence'],"%","SHAP word importance is ",pd.DataFrame(response['SHAP_word_importance']))

if __name__ == "__main__":
    main()