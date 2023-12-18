
# Sentiment Analysis with Pretrained Albert Model

The project aim to demonstrate pipelines of sentiment analysis research. In this projet use Bidirectional Encoder Representations from Transformers(BERT). BERT is Transformers based, a deep learning model in which every output element is connected to every input element, and the weightings between them are dynamically calculated based upon their connection. Use case in this project to demonstrate sentiment analysis for comments. 

## Requirements
- Python 3.10+
- Tensforflow 2.+
- PyTorch
- Scikit-learn
- Pandas
- Transformers
- Sentencepiece

## Architecture
- Albert Tokenizer
- TF Albert Sequence Classification
- Model Selection
- Tensor process
- Prediction

## How to use
1. For training models to binary sentiment classification please run `python albert_train.py`
2. For training models to multiclass sentiment classification please run `python albert_extended.py`
3. For test prediction, please run `python albert_predict.py`, make sure to config parameter models and number labels train.

## References
https://arxiv.org/abs/1810.04805 <br>
https://huggingface.co/docs/transformers/model_doc/albert