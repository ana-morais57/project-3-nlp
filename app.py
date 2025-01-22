import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    # Remove special characters, digits, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("saved_model")
    tokenizer = AutoTokenizer.from_pretrained("saved_model")
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit App Title
st.title("Fake News Detector")
st.write("Enter a headline to determine if it's real or fake news:")

# User Input
headline = st.text_area("Enter a news headline:")

if st.button("Predict"):
    if headline.strip():
        # Preprocess the headline
        processed_headline = preprocess_text(headline)
        
        # Tokenize the preprocessed input
        inputs = tokenizer(processed_headline, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=1).item()
        
        # Display results
        labels = ["Fake News", "Real News"]
        st.write(f"Prediction: **{labels[prediction]}**")
        st.write(f"Confidence: **{probs[0][prediction].item():.2f}**")
    else:
        st.warning("Please enter a valid headline!")