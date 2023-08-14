import logging

from modules.hazel_AIlib.sentiment_analyser import HazelSentimentAnalyser
from modules.hazel_AIlib.topic_modeller import HazelTopicModelAgent

import tensorflow as tf
class CONX_LEAF_ML_Pipeline():
    def meta(self, use_pre_trained):
        self.VERSION = 0.7
        self.use_pre_trained_categorizer = use_pre_trained

    def __init__(self,use_pre_trained= True) -> None:
        self.meta(use_pre_trained)
        self.init_models()
        logging.info(f"----------CONX LEAF ML Pipeline (HAZEL-AI) {self.VERSION}----------")

    def init_models(self):
        self.topic_model_agent = HazelTopicModelAgent(use_heavy_model= False)
        # self.topic_model_agent.load_sub_models()
        # self.topic_model_agent.load_model()
        self.topic_model_agent.use_pretrained_topic_modeller()

    def start_topic_modelling(self, text_content):
        clean_document = self.topic_model_agent.pre_process_text(text_content)
        topic_cluster_id = self.topic_model_agent.get_topic_name(clean_document)
        topic_category_id = self.topic_model_agent.use_pretrained_hugging_face_categorizer(clean_document)
        return (topic_cluster_id[0],topic_cluster_id[1], topic_category_id)


    def start_sentiment_analyser(self,text_content):
        sentiment_analyser_object = HazelSentimentAnalyser()
        return sentiment_analyser_object.process(text_content)
    
        
    def start_complete_text_workflow(self, text_data):
        logging.info("-> Starting Complete Text Workflow.")
        possible_topics, topic_category_name ,topic_category_id = self.start_topic_modelling(text_data)
        logging.info("-> Topic Modelling complete.")
        sentiment_dict = self.start_sentiment_analyser(text_data)
        logging.info("-> Sentiment Analyser complete.")

        response_data = {
            'topic_id': int(possible_topics),
            'topic_category_id': int(topic_category_id),
            'sentiment_value': float(sentiment_dict['sentiment_value']),
            'emotion_state': sentiment_dict['emotional_state']
        }
        return response_data


    def start_comment_text_workflow(self,text_data):
        logging.info("-> Starting Comment Text Workflow.")
        sentiment_dict = self.start_sentiment_analyser(text_data)
        response_data = {
            'sentiment_value': float(sentiment_dict['sentiment_value']),
            'emotion_state': sentiment_dict['emotional_state']
        }
        return response_data