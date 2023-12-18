import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Baca dataset dari CSV
df = pd.read_csv('sentiment_data.csv')  # Gantilah 'your_dataset.csv' sesuai dengan nama file CSV Anda

# Ubah label sentimen menjadi numerik menggunakan LabelEncoder
label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

# Bagi dataset menjadi data latih dan uji
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Tokenisasi teks
max_words = 1000

# Padding urutan token agar memiliki panjang yang sama
max_sequence_length = 20

# Model Bi-LSTM
embedding_dim = 16
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid'))
model.load_weights('bilstm.h5')


prediction = model.predict('Happy weekend!')
print(prediction)
#
