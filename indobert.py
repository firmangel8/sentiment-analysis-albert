from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Pilih model dan tokenizer IndoBERT
model_name = "indolem/indobert-base-uncased"  # Ganti dengan model yang sesuai
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Fungsi untuk melakukan sentiment analysis
def analyze_sentiment(text):
    sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    result = sentiment_analyzer(text)
    return result

# Contoh penggunaan
text_to_analyze = "Saya senang hari ini"
sentiment_result = analyze_sentiment(text_to_analyze)
print(sentiment_result)
