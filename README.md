# 📩 SMS Spam Detection using Machine Learning

This project implements a supervised machine learning pipeline for detecting spam messages from a collection of SMS messages. It demonstrates the complete process from data preprocessing and feature extraction to model training, evaluation, and prediction. This work uses the [SMS Spam Collection Dataset](https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/) from the UCI Machine Learning Repository.

## 🧠 Project Objective

The main objective of this project is to build a reliable binary classifier that can automatically categorize SMS messages as either **spam** or **ham (not spam)** using natural language processing and machine learning techniques.

---

## 📁 Repository Structure

```
.
├── spam_sms_detection.ipynb    # Jupyter notebook containing the full project code
├── SMSSpamCollection           # Raw dataset file used for training and testing
└── README.md                   # Project overview and documentation (this file)
```

---

## 🔍 Dataset Overview

* **Source**: UCI Machine Learning Repository
* **Format**: Plain text (tab-separated)
* **Size**: 5,574 SMS messages
* **Columns**:

  * `label`: `spam` or `ham`
  * `message`: the actual content of the SMS

---

## 🧰 Tools & Libraries Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* NLTK (Natural Language Toolkit)
* Matplotlib & Seaborn (for data visualization)

---

## 🚀 Workflow Summary

1. **Data Loading & Preprocessing**

   * Reading and cleaning raw text data
   * Encoding labels
   * Text normalization: lowercasing, removing stopwords, stemming

2. **Feature Extraction**

   * TF-IDF Vectorization

3. **Model Training**

   * Naive Bayes classifier (MultinomialNB)

4. **Evaluation Metrics**

   * Accuracy, Precision, Recall, F1 Score
   * Confusion Matrix

5. **Prediction on Custom Input**

   * Allows users to test custom SMS messages to check spam classification.

---

## 📊 Model Performance

* **Accuracy**: *\~98%*
* **Precision & Recall**: High precision on spam detection while maintaining strong recall on non-spam messages

---

## 💡 How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/sms-spam-detection.git
   cd sms-spam-detection
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter notebook:

   ```bash
   jupyter notebook spam_sms_detection.ipynb
   ```

---

## 📌 Future Improvements

* Incorporate deep learning models (LSTM, BERT)
* Deploy the model using a web interface or API
* Extend to multilingual spam detection

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request.

---

## 🙋‍♂️ Author

**Asmit Srivastava**
*B.Tech CSE**

