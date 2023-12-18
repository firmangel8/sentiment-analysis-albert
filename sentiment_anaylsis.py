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
train_data = tf.keras.preprocessing.sequence.pad_sequences(
    train_sequences, maxlen=max_length)
test_data = tf.keras.preprocessing.sequence.pad_sequences(
    test_sequences, maxlen=max_length)

# One-hot encode the labels
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Create the BI-LSTM model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(10000, 128, input_length=max_length))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.load_weights("sentiment_model.h5")


# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=25,
          validation_data=(test_data, test_labels))

# Save the trained model
model.save("sentiment_model.h5")
