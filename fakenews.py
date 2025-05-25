import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = {
    'text': [
        'The moon landing was a hoax.',
        'The earth is flat.',
        'Vaccines cause autism.',
        'COVID-19 is caused by 5G.',
        'The sun rises in the east.',
        'Water is wet.',
        'Python is a programming language.',
        'The sky is blue.'
    ],
    'label': ['fake', 'fake', 'fake', 'fake', 'real', 'real', 'real', 'real']
}

df = pd.DataFrame(data)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Convert text data to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)
# Train a simple Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)
# Train a simple Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)
# Predict the labels for the test set
y_pred = clf.predict(X_test_counts)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Print the confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Print the classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))


def predict_news(sentence):
    sentence_counts = vectorizer.transform([sentence])
    prediction = clf.predict(sentence_counts)
    return prediction[0]

# Example usage
new_sentence = "Earth was made by plastic."
result = predict_news(new_sentence)
print(f'The sentence "{new_sentence}" is predicted to be {result}.')




