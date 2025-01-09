import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()

# Define the text transformation function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load the pre-trained vectorizer and model
tfidf = pickle.load(open('C:\\Users\\acer\\PycharmProjects\\PythonProject\\.venv\\vectorizer.pkl', 'rb'))
model = pickle.load(open('C:\\Users\\acer\\PycharmProjects\\PythonProject\\.venv\\model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")  # Input box for the user to type a message

if st.button('Predict'):
    # 1. Preprocess the text
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize the text
    vector_input = tfidf.transform([transformed_sms])
    # 3. Make prediction
    result = model.predict(vector_input)[0]
    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
