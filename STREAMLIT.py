import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv(r"C:\Users\gokul\Downloads\spam_ham_dataset.csv")  # Update with the correct path

data['type_encoded'] = data['label'].apply(lambda x: 1 if x.lower() == 'spam' else 0)

# Preprocessing function
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply cleaning
data['cleaned_text'] = data['text'].apply(clean_text)

# Split data
X = data['cleaned_text']
y = data['type_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the individual pipelines for different models
pipeline_logreg = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000)),
    ('clf', LogisticRegression())
])

pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000)),
    ('clf', MultinomialNB())
])

pipeline_svm = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000)),
    ('clf', SVC(kernel='linear', probability=True))
])

pipeline_knn = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000)),
    ('clf', KNeighborsClassifier(n_neighbors=5))
])

# Combine pipelines using VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('logreg', pipeline_logreg),
    ('nb', pipeline_nb),
    ('svm', pipeline_svm),
    ('knn', pipeline_knn)
], voting='hard')

# Train the ensemble model
voting_clf.fit(X_train, y_train)

# Evaluate the ensemble model
y_pred = voting_clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer using pickle
with open('voting_ensemble_spam_model.pkl', 'wb') as file:
    pickle.dump(voting_clf, file)

print("Model and vectorizer saved successfully!")

import streamlit as st
import re
from nltk.corpus import stopwords

# Function to add a color theme
def add_color_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #2c2c2c;
        }
        .title h1 {
            color: #ffffff; /* White color for the title */
            font-family: 'Roboto', sans-serif;
            font-weight: bold;
        }
        .description {
            color: #ffffff;
            font-family: 'Open Sans', sans-serif;
            font-size: 18px;
        }
        .stTextArea textarea {
            background-color: #404040; /* Dark grey background for the text area */
            color: #ffffff; /* White text color */
            font-family: 'Open Sans', sans-serif;
        }
        .result-not-spam { 
            font-size: 24px; /* Large font size */ 
            font-weight: bold; /* Bold text */ 
            color: #28a745; /* Green text color */ 
            animation: fadeIn 2s;
        }
        .result-spam { 
            font-size: 24px; /* Large font size */ 
            font-weight: bold; /* Bold text */ 
            color: #dc3545; /* Red text color */
            animation: fadeIn 2s;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Load the pre-trained ensemble model
with open('voting_ensemble_spam_model.pkl', 'rb') as file:
    voting_clf = pickle.load(file)

# Preprocessing function
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit App UI
def main():
    # Add a color theme
    add_color_theme()

    # Page title
    st.markdown('<div class="title"><h1>Email Spam Detection</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Upload email content to classify it as Spam or Not Spam using an ensemble model.</div>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload an email file", type=["txt", "eml"])

    # If file is uploaded, read content
    email_text = ""
    if uploaded_file is not None:
        email_text = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded Email Content", email_text, height=200, key="input")

    # Prediction button
    if st.button("Predict"):
        if email_text.strip():
            # Clean the email content
            cleaned_email = clean_text(email_text)

            # Ensure the email text is not too short (i.e., at least two words)
            if len(cleaned_email.split()) < 2:
                st.write("Please upload a longer email message.")
            else:
                # Make prediction using the ensemble model
                final_prediction = voting_clf.predict([cleaned_email])

                # Show the final prediction result with animation
                if final_prediction[0] == 1:
                    result = "Spam"
                    st.markdown(f'<div class="result-spam">Prediction Result: {result}</div>', unsafe_allow_html=True)
                else:
                    result = "Not Spam"
                    st.markdown(f'<div class="result-not-spam">Prediction Result: {result}</div>', unsafe_allow_html=True)
        else:
            st.write("Please upload some email content.")

if __name__ == '__main__':
    main()
