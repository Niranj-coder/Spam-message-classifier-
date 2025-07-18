import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset (10 messages only)
data = {
    'message': [
        'Congratulations! You won a lottery worth ₹1,000,000!',
        'Hey, are we still meeting today?',
        'You have been selected for a cash prize. Click here.',
        'Call me when you reach home',
        'Limited offer! Claim your reward now!',
        'Don’t forget your assignment is due today.',
        'Get free recharge now!',
        'Let’s catch up sometime soon.',
        'Winner! Claim your ₹50000 gift card now!',
        'Are you coming to class tomorrow?'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

# Load data
df = pd.DataFrame(data)

# Preprocess
X = df['message']
y = df['label']

# Convert text to numbers
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

# Split for testing
X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Predict your own message
while True:
    msg = input("\n📨 Enter an SMS (or type 'quit' to exit):\n> ")
    if msg.lower() == "quit":
        break
    vec = vectorizer.transform([msg])
    result = model.predict(vec)
    print(f"🔍 Prediction: {'🚫 SPAM' if result[0] == 'spam' else '✅ NOT SPAM'}")