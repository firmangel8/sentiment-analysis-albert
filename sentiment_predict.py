import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
# Load the dataset
data = pd.read_csv("sentiment_data.csv")

# Split the data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data['text'], data['sentiment'], test_size=0.2)

# Tokenize the text and create a vocabulary
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data)
train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

# Pad the sequences to the same length
max_length = max([len(s) for s in train_sequences + test_sequences])

# Use the trained model to make predictions on new data
model = tf.keras.Sequential()
# model = tf.keras.models.load_model("sentiment_model.h5")
model.add(tf.keras.layers.Embedding(10000, 128, input_length=max_length))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.load_weights("sentiment_model.h5")

# new_data = ["This movie is terrible", "I not like this movie", "The book was great"]
new_data = ["Delightful surprise!"]
new_sequences = tokenizer.texts_to_sequences(new_data)
print(new_sequences)
new_data = tf.keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=max_length)
predictions = model.predict(new_data)
threshold = 0.5

for pred in predictions:
    print(pred)
    if(pred[0]> threshold):
        print('Positive')
    elif(pred[0] < threshold):
        print('Negative')
    else:
        print('Neutral')

