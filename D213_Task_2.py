import pandas as pd
import zipfile
import io
import nltk
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Download stopwords
nltk.download('stopwords')

# Load and concatenate datasets
zip_path = "sentiment_labelled_sentences.zip"
zipped_txts = {
    "sentiment labelled sentences/amazon_cells_labelled.txt": "amzn",
    "sentiment labelled sentences/imdb_labelled.txt": "imdb",
    "sentiment labelled sentences/yelp_labelled.txt": "yelp"
}
dfs = {}
with zipfile.ZipFile(zip_path, 'r') as z:
    for file_path, df_name in zipped_txts.items():
        with z.open(file_path) as f:
            dfs[df_name] = pd.read_csv(io.TextIOWrapper(f), sep="\t", header=None, names=["sentence", "label"])

# Map source
source_mapping = {"amzn": 1, "imdb": 2, "yelp": 3}
dfs_list = []
for df_name, source_value in source_mapping.items():
    df = dfs[df_name].copy()
    df["source"] = source_value
    dfs_list.append(df)
concatenated_data = pd.concat(dfs_list, ignore_index=True)

# Clean and normalize text
concatenated_data['sentence'] = concatenated_data['sentence'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
concatenated_data['sentence'] = concatenated_data['sentence'].str.lower()

# Visualize sentiment distribution
label_counts = concatenated_data['label'].value_counts()
fig = go.Figure(data=[
    go.Bar(x=label_counts.index, y=label_counts.values, text=label_counts.values,
           textposition='auto', marker_color=['blue', 'red'])
])
fig.update_layout(title='Sentiment Distribution', xaxis_title='Sentiment', yaxis_title='Count')
fig.show()

# Remove stopwords
stop_words = set(stopwords.words('english'))
concatenated_data['sentence'] = concatenated_data['sentence'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

# Export preprocessed data
concatenated_data.to_csv("prepared_data.csv", index=False)

# Split train/test
split = int(len(concatenated_data) * 0.8)
train = concatenated_data[:split]
test = concatenated_data[split:]

# Tokenization and padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['sentence'])
train_sequences = tokenizer.texts_to_sequences(train['sentence'])
test_sequences = tokenizer.texts_to_sequences(test['sentence'])
maxlen = 50
train_padded = pad_sequences(train_sequences, maxlen=maxlen, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=maxlen, padding='post', truncating='post')

# Build model
VOCAB_SIZE = len(tokenizer.word_index) + 1
EMBED_DIM = 16
model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM, input_length=maxlen),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(
    train_padded,
    train['label'].values,
    epochs=100,
    validation_data=(test_padded, test['label'].values),
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate model
loss, accuracy = model.evaluate(test_padded, test['label'].values, verbose=1)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
fig = make_subplots(rows=1, cols=2,
    subplot_titles=('Training and Validation Accuracy', 'Training and Validation Loss'))
fig.add_trace(go.Scatter(x=list(epochs), y=acc, mode='lines', name='Training Accuracy'), row=1, col=1)
fig.add_trace(go.Scatter(x=list(epochs), y=val_acc, mode='lines', name='Validation Accuracy'), row=1, col=1)
fig.add_trace(go.Scatter(x=list(epochs), y=loss, mode='lines', name='Training Loss'), row=1, col=2)
fig.add_trace(go.Scatter(x=list(epochs), y=val_loss, mode='lines', name='Validation Loss'), row=1, col=2)
fig.update_xaxes(title_text='Epochs', row=1, col=1)
fig.update_yaxes(title_text='Accuracy', row=1, col=1)
fig.update_xaxes(title_text='Epochs', row=1, col=2)
fig.update_yaxes(title_text='Loss', row=1, col=2)
fig.show()

# Predictions
predictions = model.predict(test_padded)
predicted_labels = [1 if p > 0.5 else 0 for p in predictions]

# Evaluation
print("\nSample Predictions:\n")
for sentence, label, predicted_label in zip(test['sentence'].head(10), test['label'].head(10), predicted_labels[:10]):
    print(f"Sentence: {sentence}")
    print(f"Actual Label: {label}")
    print(f"Predicted Label: {predicted_label}\n")

print("\nClassification Report:\n")
print(classification_report(test['label'], predicted_labels))
print("\nConfusion Matrix:\n")
print(confusion_matrix(test['label'], predicted_labels))

# Save and reload model
model.save('sentiment_analysis_model.h5')
loaded_model = load_model('sentiment_analysis_model.h5')