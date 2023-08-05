import logging
from transformers import pipeline


class HazelSentimentAnalyser():

    def meta(self):
        self.VERSION = 0.5
    

    def __init__(self):
        self.meta()
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.emotion_pipeline = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion')
        logging.info(f"Hazel Sentiment Analyser {self.VERSION}")


    def generate_sentiment_value(self,text):
        return self.sentiment_pipeline([text])


    def generate_emotion_state(self, text):
        return self.emotion_pipeline(text)


    def process(self,text):
        sentiment_value = self.generate_sentiment_value(text)[0]
        emotional_state = self.generate_emotion_state(text)[0]
        if sentiment_value['label'] == "NEGATIVE":
            sentiment_percentage = -1 * sentiment_value['score']
        else:
            sentiment_percentage = sentiment_value['score']

        sentiment_dict = {
            'sentiment_value': sentiment_percentage,
            'emotional_state': emotional_state['label']
        }

        return sentiment_dict
