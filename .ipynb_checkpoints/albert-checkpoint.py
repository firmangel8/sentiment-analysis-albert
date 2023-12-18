import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
import tensorflow as tf

# Hyperparameter configuration
EPOCH = 100
BATCH = 32
LEARNING_RATE = 1e-5

# Baca dataset dari CSV
file_path = 'sentiment_data_albert.csv'  
df = pd.read_csv(file_path)

# Persiapkan data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].values,
    df['sentiment'].values,
    test_size=0.2,
    random_state=42
)

# Unduh ALBERT Pre-trained Model
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2')

# Preprocessing Data
max_length = 1000

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=max_length, return_tensors='tf')
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=max_length, return_tensors='tf')

# Ekstrak Array NumPy
train_input_ids = train_encodings['input_ids'].numpy()
train_attention_mask = train_encodings['attention_mask'].numpy()

test_input_ids = test_encodings['input_ids'].numpy()
test_attention_mask = test_encodings['attention_mask'].numpy()

# Konversi label sentimen menjadi bentuk numerik
label_mapping = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
train_labels_numeric = [label_mapping[label] for label in train_labels]
test_labels_numeric = [label_mapping[label] for label in test_labels]

# Pastikan bahwa label yang dihasilkan sesuai dengan rentang model
train_labels_numeric = [label if label in [0, 1, 2] else 0 for label in train_labels_numeric]
test_labels_numeric = [label if label in [0, 1, 2] else 0 for label in test_labels_numeric]


# Buat tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(((train_input_ids, train_attention_mask), train_labels_numeric))
test_dataset = tf.data.Dataset.from_tensor_slices(((test_input_ids, test_attention_mask), test_labels_numeric))

# Training Model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset.batch(BATCH), epochs=EPOCH)


# Evaluasi Model
eval_results = model.evaluate(test_dataset.batch(BATCH))
print("Test loss:", eval_results[0])
print("Test accuracy:", eval_results[1])

# Prediksi dengan Model yang Telah Dilatih
new_texts = ['The books are awesome', 'Nice to meet you', 'So sad to hear the news']
new_encodings = tokenizer(new_texts, truncation=True, padding=True, max_length=max_length, return_tensors='tf')

new_input_ids = new_encodings['input_ids'].numpy()
new_attention_mask = new_encodings['attention_mask'].numpy()

predictions = model.predict([new_input_ids, new_attention_mask])
# Mengambil logits dari TFSequenceClassifierOutput
logits = predictions.logits

# Mengambil prediksi sentimen
predicted_labels = tf.argmax(logits, axis=1).numpy()
predicted_sentiments = [list(label_mapping.keys())[list(label_mapping.values()).index(label)] for label in predicted_labels]
print("Predicted sentiments:", predicted_sentiments)

model.save_weights('my-albert.h5')
