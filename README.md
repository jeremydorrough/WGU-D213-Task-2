# WGU D213 Task 2: Sentiment Analysis Using Neural Networks and NLP

This project demonstrates a complete end-to-end sentiment analysis pipeline using Natural Language Processing (NLP) and a neural network built with TensorFlow/Keras. It processes and classifies customer reviews from Amazon, Yelp, and IMDB datasets to predict positive or negative sentiment.

## 🎯 Objective

**Business Question:**  
Can customer reviews be analyzed using neural networks and NLP to accurately predict their sentiment (like/dislike), helping organizations improve products and services?

**Goal:**  
Train a binary classification model to classify customer sentiment using a lightweight neural network with embedding, pooling, and dense layers.

## 📁 Files

- `PA_Jeremy_Dorrough_D213_Task_2-combined.docx`: Full written report with explanations of methodology, design, and results
- `prepared_data.csv`: Cleaned and preprocessed dataset
- `sentiment_analysis_model.h5`: Trained TensorFlow model

## 🧪 Tools and Libraries

- Python 3.x
- `pandas`, `numpy`, `nltk` – Data processing and cleaning
- `TensorFlow`, `Keras` – Deep learning framework
- `sklearn` – Metrics and evaluation
- `plotly` – Interactive visualizations

## 🧹 Preprocessing

- Combined labeled datasets from Amazon, IMDB, and Yelp
- Removed special characters, converted to lowercase, and removed stopwords
- Tokenized text using Keras' `Tokenizer`
- Padded sequences to uniform length of 50
- Exported the final preprocessed dataset for reuse

## 🔢 Model Architecture

- **Embedding Layer**: 16-dimension vector for each word
- **GlobalAveragePooling1D**: Reduces tensor dimensionality
- **Dense Layer (ReLU)**: 16 nodes for nonlinear feature learning
- **Dense Output (Sigmoid)**: Single node for binary classification

**Training Details:**
- Loss Function: `binary_crossentropy`
- Optimizer: `adam`
- Early stopping: Monitored validation loss with 2-epoch patience
- Epochs: 100 max, early stopped at epoch 12

## 📈 Results

- **Test Accuracy:** 79.09%
- **Precision (Negative):** 0.84
- **Recall (Negative):** 0.77
- **Precision (Positive):** 0.74
- **Recall (Positive):** 0.81
- **Confusion Matrix:**

[[231 70]
[ 48 201]]

- **Observations:**
- Balanced performance between positive and negative classes
- Some misclassifications indicate room for improvement (false positives/negatives)
- Overfitting controlled via early stopping and regularization

## 📌 Insights

- The most frequent words in positive reviews: `good`, `great`, `best`, `love`
- The model effectively learns overall sentiment without being biased toward specific words
- Simple, interpretable architecture makes it deployable in resource-constrained environments

## 💡 Recommendations

- Further improve performance with more data and hyperparameter tuning
- Add dropout layers or experiment with LSTM/GRU for context understanding
- Deploy as a lightweight API for real-time review analysis in customer service applications

## 👤 Author

Jeremy Dorrough  
Western Governors University  

## 📚 References

- [TensorFlow Embeddings](https://www.tensorflow.org/text/guide/word_embeddings)
- [Machine Learning Mastery – Model Training History](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)
- [GeeksForGeeks – Removing Stopwords](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/)
- [CS50 AI – NLP](https://cs50.harvard.edu/ai/2020/notes/5/)
- [AskPython – Predict Function](https://www.askpython.com/python/examples/python-predict-function)
