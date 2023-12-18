from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
import tensorflow as tf

# Download ALBERT Pre-trained Model
label_mapping = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
# label_mapping = {'Very Positive': 0, 'Very Negative': 1, 'Mixed': 2, 'Positive': 3, 'Negative': 4, 'Neutral': 5}
MAX_LENGTH = 1000
NUM_LABELS = 2
MODEL_PATH = 'my-albert-202312141619.h5'


tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=NUM_LABELS)  # Adjust num_labels based on the number of sentiments
model.load_weights(MODEL_PATH)

new_texts = ['Excited about the upcoming features!', 'Disappointed with the latest changes.']
new_encodings = tokenizer(new_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='tf')

new_input_ids = new_encodings['input_ids'].numpy()
new_attention_mask = new_encodings['attention_mask'].numpy()


# Mengambil logits dari TFSequenceClassifierOutput dan lakukan predictions
predictions = model.predict([new_input_ids, new_attention_mask])
logits = predictions.logits
predicted_labels = tf.argmax(logits, axis=1).numpy()
predicted_sentiments = [list(label_mapping.keys())[list(label_mapping.values()).index(label)] for label in predicted_labels]
print("Predicted sentiments:", predicted_sentiments)