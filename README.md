This is an educational text. The consequences are your own responsibility.

# web-text-classifier
"A simple Python web-based text classifier using Naive Bayes and TF-IDF.

```
# Install required libraries if needed:
# pip install requests beautifulsoup4 scikit-learn nltk

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords

# Download stopwords for the first time
nltk.download('stopwords')

# ------------- Step 1: Data Collection ----------------
def get_website_text(url):
    # Add User-Agent header to avoid request blocks
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract all text from the page
        text = soup.get_text(separator=' ')
        return text
    else:
        print("Error fetching the URL:", url, "Status code:", response.status_code)
        return ""

# Example: Collect information from a website
keyword = "Python"
url = f"https://en.wikipedia.org/wiki/{keyword}"
text_data = get_website_text(url)

# ----------------- Step 2: Text Preprocessing -----------------
stop_words = set(stopwords.words('english'))
# Keep only alphabetic words and remove stopwords
words = [word.lower() for word in text_data.split() if word.isalpha() and word.lower() not in stop_words]
processed_text = ' '.join(words)

# ----------------- Step 3: Prepare Data for ML -----------------
# Example: creating a simple dataset with labels
texts = [processed_text, "Python is a programming language.", "Machine learning is fun.", "I love coding."]
labels = ["wiki", "education", "education", "hobby"]

# Convert texts into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------- Step 4: Train Machine Learning Model -----------------
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ----------------- Step 5: Use the Model for New Text -----------------
new_text = "Python is widely used for AI and ML projects."
new_vector = vectorizer.transform([new_text])
prediction = model.predict(new_vector)
print("Prediction for new text:", prediction[0])
```

A Python project that scrapes text from websites, preprocesses it, and classifies it using a simple Naive Bayes machine learning model.

Detailed Explanation:
This project demonstrates a complete workflow for text classification in Python:

Data Collection:

Uses the requests library and BeautifulSoup to fetch text content from a given URL.

A User-Agent header is added to avoid request blocks.

Text Preprocessing:

Removes stopwords using NLTK’s English stopwords list.

Keeps only alphabetic words and converts all text to lowercase.

Dataset Preparation:

The scraped text is combined with sample texts to create a simple labeled dataset.

Texts are converted into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency).

Model Training:

A Multinomial Naive Bayes classifier is trained on the dataset.

The model is evaluated on a test split, and the accuracy is printed.

Prediction:

The trained model can predict categories for new texts.

Example: classifying a sentence like "Python is widely used for AI and ML projects."

Libraries Used:

requests – for web scraping

beautifulsoup4 – for parsing HTML

scikit-learn – for feature extraction, model training, and evaluation

nltk – for natural language processing (stopwords removal)

Usage Example:

pip install requests beautifulsoup4 scikit-learn nltk
python text_classifier.py

Notes:

This is an educational project to demonstrate web scraping and basic text classification.

You can replace the sample dataset with your own labeled texts to improve predictions.

TF-IDF and Naive Bayes provide a simple but effective baseline for text categorization tasks.


