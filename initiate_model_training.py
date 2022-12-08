from LSTM import SentimentAnalysis_LSTM
from BERT import BERT_ModelTraining
from os.path import exists

if __name__ == '__main__':

    # LSTM Model Train
    Lstm_exists = exists("model_weight/LSTM_Model.hdf5")
    if not Lstm_exists:
        SentimentAnalysis_LSTM.train_predict('train')

    # BERT Model Train
    Bert_exists = exists("model_weight/BERT_Model.hdf5")
    if not Bert_exists:
        BERT_ModelTraining.SentimentAnalyser()
