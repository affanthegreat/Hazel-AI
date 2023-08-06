import logging
from transformers import pipeline


class HazelSentimentAnalyser():

    def meta(self):
        self.VERSION = 0.5
    

    def __init__(self):
        self.meta()
        self.text_sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.text_sentiment_pipeline = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion')
        self.image_to_text_pipeline = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        logging.info(f"Hazel Sentiment Analyser {self.VERSION}")


    def generate_sentiment_value(self,text):
        return self.text_sentiment_pipeline([text])


    def generate_emotion_state(self, text):
        return self.text_sentiment_pipeline(text)

    def generate_image_caption(self, image_path):
        return self.image_to_text_pipeline(image_path)

    def image_process(self,image_path):
        image_caption = self.generate_image_caption(image_path)[0]['generated_text']
        sentiment_dict = self.process(image_caption)
        return sentiment_dict
    
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
    
   