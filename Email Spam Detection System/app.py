import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained model and CountVectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit application
st.title("Email Spam Classifier")

# Text input for the message
message = st.text_area("Type an Email here to check it is Spam or Ham", height=150)

if st.button("Predict"):
    # Preprocess and vectorize the input message
    message_vectorized = vectorizer.transform([message])
    
    # Predict using the loaded model
    prediction = model.predict(message_vectorized)
    
    # Display the result
    if prediction == 1:
        st.error("warning: This is a spam message.")
    else:
        st.success("Great: This is NOT a spam message.")
