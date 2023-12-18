""" Train model for multiclass sentiment analysis """
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
import tensorflow as tf

from albert_predict import NUM_LABELS

# Read dataset from CSV
FILE_PATH = 'sentiment_data_extended.csv'
df = pd.read_csv(FILE_PATH)
EPOCH = 100
BATCH = 32
LEARNING_RATE = 1e-5
MAX_LENGTH = 128
BASE_PRETRAINED_MODEL='albert-base-v2'
NUM_LABELS=6

# Prepare data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].values,
    df['sentiment'].values,
    test_size=0.2,
    random_state=42
)

# Download ALBERT Pre-trained Model
tokenizer = AlbertTokenizer.from_pretrained(BASE_PRETRAINED_MODEL)
model = TFAlbertForSequenceClassification.from_pretrained(BASE_PRETRAINED_MODEL, num_labels=NUM_LABELS)  # Adjust num_labels based on the number of sentiments



train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='tf')
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='tf')

# Extract NumPy arrays from BatchEncoding
train_input_ids = train_encodings['input_ids'].numpy()
train_attention_mask = train_encodings['attention_mask'].numpy()

test_input_ids = test_encodings['input_ids'].numpy()
test_attention_mask = test_encodings['attention_mask'].numpy()

# Convert sentiment labels to numeric format
label_mapping = {'Very Positive': 0, 'Very Negative': 1, 'Mixed': 2, 'Positive': 3, 'Negative': 4, 'Neutral': 5}
train_labels_numeric = [label_mapping[label] for label in train_labels]
test_labels_numeric = [label_mapping[label] for label in test_labels]

# Create tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(((train_input_ids, train_attention_mask), train_labels_numeric))
test_dataset = tf.data.Dataset.from_tensor_slices(((test_input_ids, test_attention_mask), test_labels_numeric))

# Training Model
# pylint: disable=no-member
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE) # type: ignore
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']) # type: ignore

model.fit(train_dataset.batch(BATCH), epochs=EPOCH) # type: ignore

# Evaluate Model
eval_results = model.evaluate(test_dataset.batch(BATCH)) # type: ignore
print("Test loss:", eval_results[0])
print("Test accuracy:", eval_results[1])

# Predict with the Trained Model
new_texts = ['Excited about the upcoming features!', 'Disappointed with the latest changes.']
new_encodings = tokenizer(new_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='tf')

new_input_ids = new_encodings['input_ids'].numpy()
new_attention_mask = new_encodings['attention_mask'].numpy()

predictions = model.predict([new_input_ids, new_attention_mask]) # type: ignore
# Mengambil logits dari TFSequenceClassifierOutput
logits = predictions.logits
# Mengambil prediksi sentimen
predicted_labels = tf.argmax(logits, axis=1).numpy()
predicted_sentiments = [list(label_mapping.keys())[list(label_mapping.values()).index(label)] for label in predicted_labels]
print("Predicted sentiments:", predicted_sentiments)

model.save_weights('albert-extended.h5') # type: ignore
