import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to scrape Wikipedia page
def scrape_wikipedia(country):
    url = f"https://en.wikipedia.org/wiki/Brazil"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

# Function to clean text
def clean_text(text):
    text = re.sub(r'\[[0-9]*\]', '', text)
    text = re.sub(r'[^a-zA-Z. ]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Scrape and preprocess text
text = scrape_wikipedia("Brazil")
cleaned_text = clean_text(text)
sentences = sent_tokenize(cleaned_text)

# Sentiment Analysis
sentiments = [TextBlob(sent).sentiment.polarity for sent in sentences]
sentiment_labels = ['Positive' if s > 0 else 'Negative' if s < 0 else 'Neutral' for s in sentiments]

# Create DataFrame
df = pd.DataFrame({'Sentence': sentences, 'Sentiment': sentiment_labels})

# Remove Neutral Sentences
df = df[df['Sentiment'] != 'Neutral']

# Word Tokenization and Stopword Removal
stop_words = set(stopwords.words('english'))
words = word_tokenize(cleaned_text)
filtered_words = [word for word in words if word not in stop_words]

# WordCloud Generation
def generate_wordcloud():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Sentence'])
y = df['Sentiment'].map({'Positive': 1, 'Negative': 0})

# Balance Sentiment Data Using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train ML Models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Naïve Bayes': MultinomialNB(),
    'KNN': KNeighborsClassifier()
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    trained_models[name] = model
    pickle.dump(model, open(f"{name}.pkl", "wb"))

# Streamlit UI
st.title("Sentiment Analysis on Brazil Wikipedia Data")

st.write("### WordCloud")
generate_wordcloud()

st.write("### Input Sentence for Sentiment Prediction")
user_input = st.text_input("Enter a sentence:")
model_choice = st.selectbox("Choose a Model", list(models.keys()))

if st.button("Predict Sentiment"):
    model = pickle.load(open(f"{model_choice}.pkl", "rb"))
    input_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(input_tfidf)[0]
    sentiment_result = "Positive" if prediction == 1 else "Negative"
    st.write(f"Predicted Sentiment: **{sentiment_result}**")
