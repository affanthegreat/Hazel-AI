import logging

from modules.hazel_AIlib.sentiment_analyser import HazelSentimentAnalyser
from modules.hazel_AIlib.topic_modeller import HazelTopicModelAgent

class CONX_LEAF_ML_Pipeline():
    def meta(self):
        self.VERSION = 0.6

    def __init__(self) -> None:
        self.meta()
        self.init_models()
        logging.info(f"----------CONX LEAF ML Pipeline (HAZEL-AI) {self.VERSION}----------")

    def init_models(self):
        self.topic_model_agent = HazelTopicModelAgent(use_heavy_model= True)
        self.topic_model_agent.load_sub_models()
        self.topic_model_agent.load_model()

    def start_topic_modelling(self, text_content):
        clean_document = self.topic_model_agent.pre_process_text(text_content)
        topic_cluster_id, cluster_name = self.topic_model_agent.get_topic_name(clean_document)
        return topic_cluster_id, cluster_name


    def start_sentiment_analyser(self,text_content):
        sentiment_analyser_object = HazelSentimentAnalyser()
        return sentiment_analyser_object.process(text_content)
    
        
    def start_complete_text_workflow(self, text_data):
        logging.info("-> Starting Complete Text Workflow.")
        possible_topics, cluster_names = self.start_topic_modelling(text_data)
        logging.info("-> Topic Modelling complete.")
        sentiment_dict = self.start_sentiment_analyser(text_data)
        logging.info("-> Sentiment Analyser complete.")

        response_data = {
            'topic_id': int(possible_topics),
            'cluster_name': cluster_names[0],
            'sentiment_value': int(sentiment_dict['sentiment_value']),
            'emotion_state': sentiment_dict['emotional_state']
        }
        return response_data


    def start_comment_text_workflow(self,text_data):
        logging.info("-> Starting Comment Text Workflow.")
        sentiment_dict = self.start_sentiment_analyser(text_data)
        response_data = {
            'sentiment_value': int(sentiment_dict['sentiment_value']),
            'emotion_state': sentiment_dict['emotional_state']
        }
        return response_data