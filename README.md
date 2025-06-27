# 📧 Email Spam Classifier

A machine learning-based classifier that detects whether an email is **spam** or **not spam (ham)**. It uses natural language processing (NLP) techniques to clean, vectorize, and classify email texts using various models.

---

## 🚀 Features

- 🧹 Text preprocessing: stopword removal, stemming, tokenization
- 🔢 Text vectorization using **TF-IDF**
- 🤖 ML models: Naive Bayes (default), SVM, Logistic Regression
- 📊 Accuracy metrics and confusion matrix
- 🌐 Web UI using **Streamlit** (optional)
- 🧪 Easily extendable to new datasets

---

## 🧠 Tech Stack

- Python 3.x
- scikit-learn
- pandas
- numpy
- NLTK
- Streamlit (optional for UI)

---

## 📁 Folder Structure

- `app.py` – (Optional) Streamlit app to interact with the classifier
- `spam_classifier.py` – Main classification script
- `preprocessing.py` – Text cleaning and transformation logic
- `vectorizer.pkl` – Saved TF-IDF vectorizer (optional)
- `model.pkl` – Trained model saved via joblib/pickle
- `emails.csv` – Dataset (e.g., SpamAssassin or custom CSV)

---

## ✅ How to Run

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/email-spam-classifier.git
   cd email-spam-classifier
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run classifier (Jupyter or script):

bash
Copy
Edit
python spam_classifier.py
(Optional) Run Streamlit app:

bash
Copy
Edit
streamlit run app.py
📈 Example Output
vbnet
Copy
Edit
Input: "Congratulations! You’ve won a $1000 Walmart gift card. Click here."
Prediction: SPAM ✅

Input: "Can we reschedule our meeting to tomorrow morning?"
Prediction: NOT SPAM ✅
📊 Evaluation Metrics
Accuracy: ~98%

Precision, Recall, F1-score

Confusion matrix visualized using seaborn/matplotlib

📌 Dataset Used
You can use:

SMS Spam Collection Dataset (UCI)

Or your own custom email/spam dataset in .csv format

🙋‍♂️ Author
Made with 💻 by Priyansh Bansal
GitHub Profile

📄 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

Let me know if:
- You want a version **without Streamlit**
- You're using a **different dataset**
- You want a short LinkedIn summary or project description too

I can also generate `requirements.txt` or a minimal working `app.py` for it if needed.
