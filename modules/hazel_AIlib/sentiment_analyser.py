import logging
from transformers import pipeline


class HazelSentimentAnalyser():

    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.emotion_pipeline = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion')
        logging.info("Hazel Sentiment Analyser v1")

    def generate_sentiment_value(self,text):
        self.sentiment_pipeline([text])

    def generate_emotion_state(self, text):
        self.emotion_pipeline(text)


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
