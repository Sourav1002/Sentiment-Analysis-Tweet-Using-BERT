from flask import Flask, request, render_template
from Vader_TextBlob import TextBlobModel, VaderSentimentModel
from BERT import BERT_ModelPredict
from LSTM import SentimentAnalysis_LSTM

app = Flask(__name__)
app.secret_key = "secret key"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/modelexec', methods=['GET', 'POST'])
def model_execution():
    if request.method == 'GET':
        input_Text = request.args.get("input_Text")
        print(input_Text)
        result = {
            "textblob": TextBlobModel.TextBlob_Predict(input_Text),
            "vader": VaderSentimentModel.VaderSentiment_Predict(input_Text),
            "bert": BERT_ModelPredict.predict(input_Text),
            "lstm": SentimentAnalysis_LSTM.train_predict('test', input_Text)
        }
        return result


if __name__ == '__main__':
    app.run(debug=True)
