# IMDB Sentiment Analysis with SimpleRNN

This project performs sentiment analysis on the IMDB dataset using a Recurrent Neural Network (RNN) implemented with Keras. Below is a summary of the project setup, steps, and results.

## Project Structure

- **Dataset**: The IMDB movie reviews dataset, included with Keras.
- **Exploratory Data Analysis (EDA)**: Visualization of class distribution and review lengths.
- **Preprocessing**: Padding sequences to ensure uniform input size.
- **Model Architecture**: SimpleRNN with embedding, Dense, and Activation layers.
- **Training and Evaluation**: Model training and validation using accuracy and loss metrics.

---

## Steps

### 1. Data Loading

The IMDB dataset is loaded using Keras with the following parameters:
- `num_words`: Limits the vocabulary size to 15,000.
- `maxlen`: Limits the review length to 130 tokens using padding.

```python
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

num_words = 15000
maxlen = 130
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=num_words)
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
```

---

### 2. Exploratory Data Analysis (EDA)

- **Label Distribution**: The dataset is balanced with 50% positive (1) and 50% negative (0) labels.
- **Visualization**:
  - Class frequencies plotted using Seaborn.
  - Review lengths visualized with distribution plots.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(Y_train)
plt.title("Y_train Distribution")
plt.show()

sns.histplot([len(review) for review in X_train], kde=True, label="Train")
sns.histplot([len(review) for review in X_test], kde=True, label="Test")
plt.title("Review Lengths")
plt.legend()
plt.show()
```

---

### 3. Model Architecture

The model consists of:
- An **Embedding Layer**: Converts word indices into dense vectors.
- A **SimpleRNN Layer**: Processes sequences with 16 units.
- A **Dense Layer**: Outputs a single value for binary classification.
- A **Sigmoid Activation Layer**: For binary output probabilities.

```python
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, Activation

rnn = Sequential([
    Embedding(num_words, 32, input_length=maxlen),
    SimpleRNN(16, activation="relu"),
    Dense(1),
    Activation("sigmoid")
])

rnn.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
```

---

### 4. Training and Evaluation

The model is trained for 5 epochs with a batch size of 128. Training accuracy, validation accuracy, and loss are monitored.

```python
history = rnn.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=128, verbose=1)

score = rnn.evaluate(X_test, Y_test)
print(f"Accuracy: {score[1] * 100:.2f}%")
```

- Final Test Accuracy: **85.92%**

---

### 5. Results Visualization

Accuracy and loss plots for training and validation data:

```python
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.legend()
plt.show()
```

---

## Dependencies

- Python
- Keras
- TensorFlow
- NumPy
- Matplotlib
- Seaborn

---

## Usage

1. Install the required libraries:
   ```bash
   pip install numpy matplotlib seaborn tensorflow keras
   ```

2. Run the script to train and evaluate the model.

---

## Notes

- The model can be further improved with techniques like LSTMs, GRUs, or hyperparameter tuning.
- Replace deprecated `distplot` with `histplot` for future Seaborn versions.
