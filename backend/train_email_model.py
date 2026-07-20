import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv('datasets/enron_spam_data.csv')

print("Dataset loaded. Total emails:", len(data))
print(data['Spam/Ham'].value_counts())

# Drop rows with missing message text
data = data.dropna(subset=['Message'])

# Convert labels: ham = 0 (safe), spam = 1 (scam)
data['label_num'] = data['Spam/Ham'].map({'ham': 0, 'spam': 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['Message'], data['label_num'], test_size=0.2, random_state=42
)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test accuracy
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and vectorizer
with open('models/email_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/email_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Email model and vectorizer saved successfully!")
