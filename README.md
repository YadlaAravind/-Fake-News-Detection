# 📰 Fake News Detection using Naive Bayes

A simple yet effective project to detect fake news using text classification techniques in Python. It utilizes a small set of example statements to train a Naive Bayes model and predict whether a given sentence is **real** or **fake**.

---

## 📌 Project Highlights

- ✅ Binary classification: real vs. fake
- 🧠 Model: Multinomial Naive Bayes
- 🗂 Data Vectorization: CountVectorizer
- 📊 Evaluation: Accuracy, confusion matrix, and classification report
- 🧪 Predict new input sentences

---

## 📂 Dataset

The dataset is manually created in the script:

```plaintext
"The moon landing was a hoax." → fake  
"Python is a programming language." → real  
...
```
🚀 How It Works
Vectorizes text using CountVectorizer

Splits into train/test using train_test_split

Trains a MultinomialNB model

Evaluates with accuracy_score and confusion_matrix

Predicts a new sentence

▶️ Example Usage
Run the script:
```
python fakenews.py
```
Expected output:
```
Accuracy: 1.0
Confusion Matrix:
[[1 0]
 [0 1]]
Classification Report:
              precision    recall  f1-score   support
         fake       1.00      1.00      1.00         1
         real       1.00      1.00      1.00         1
The sentence "Earth was made by plastic." is predicted to be fake.
```
📦 Requirements:
-Python 3.x

-pandas

-scikit-learn

Install dependencies:
```
pip install pandas scikit-learn

```
👨‍💻 Author
Developed by Yadla Aravind
