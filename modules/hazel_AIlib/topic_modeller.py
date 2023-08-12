import os
import logging


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset

import numpy as np
from scipy.special import expit
import tensorflow_hub

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer


from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

class HazelTopicModelAgent():

    def get_agent_info(self):
        self.VERSION = 0.7
        self.BUILD = "STABLE"
        self.model = None
        self.models_available = ['2Mil_C4', "4Mil_C8_Heavy"]

    def init_ntlk(self):
        logging.info("Intializing NLTK.")
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')

    def __init__(self, use_heavy_model= False, use_pre_trained=True):

        self.init_ntlk()
        self.get_agent_info()
        logging.info(f"====== EDEN TOPIC MODELLER V{self.VERSION} {self.BUILD} ======")
        self.use_pre_trained = use_pre_trained
        if use_heavy_model:
            self.model_type = "4Mil_C8_Heavy"
        else:
            self.model_type = "2Mil_C4"

    def load_sub_models(self):
        self.embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        #self.embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
 
        self.umap_model = IncrementalPCA(n_components=256)
        self.cluster_model = MiniBatchKMeans(n_clusters=1024, random_state=0)
        self.vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=.01)
        self.representation_model = {
            "KeyBERT": KeyBERTInspired(),
            "MMR": MaximalMarginalRelevance(diversity=0.6)

        }
        logging.info("-> Default sub-models loaded. ")
    
        
    def predict_category(self, text):
        self.pre_trained_model_name = f"cardiffnlp/tweet-topic-21-multi"
        self.pre_trained_model = AutoModelForSequenceClassification.from_pretrained(self.pre_trained_model_name, low_cpu_mem_usage=True)
        self.pre_trained_tokenizer = AutoTokenizer.from_pretrained(self.pre_trained_model_name)
        print(text)
        tokens = self.pre_trained_tokenizer(text, return_tensors='pt')
        output = self.pre_trained_model(**tokens)
        return output
        
    def clean_output_pre_trained(self,output):
        try:
            class_mapping = self.pre_trained_model.config.id2label
            scores = expit(output[0][0].detach().numpy())
            topic_category_id, topic_category_id_score = sorted({category:score for score,category in zip(scores, class_mapping) 
                        if score > 0.5}.items(), key=lambda x:x[1])[-1]
            
            return topic_category_id
        except IndexError:
            #19 refers to OTHER / Unclassified.
            topic_category_id = 19
            topic_category_id_score = 100
            return topic_category_id

    def use_pretrained_hugging_face_categorizer(self,text):
        unclean_output =self.predict_category(text)
        topic_category_id = self.clean_output_pre_trained(unclean_output)
        return topic_category_id


    def get_new_model_instance(self):

        topic_model = BERTopic(min_topic_size=60,
                               embedding_model=self.embedding_model,
                               vectorizer_model=self.vectorizer_model,
                               representation_model=self.representation_model,
                               hdbscan_model=self.cluster_model,
                               verbose=True)
        return topic_model
    
    def load_categorizer(self):
        if not self.use_pre_trained:
            model_path = f'topic_models/categorizer'
            if os.path.exists(model_path):
                self.model = BERTopic.load(model_path,embedding_model= self.embedding_model)
                return 100
            else:
                logging.error(f"> Categorizer Model not found in the default path. Check whether saved model exists at {model_path}")
                return 200
        else:
            logging.error(f"> Pre-trained model is being used with the current instance of the agent. try passing 'use_pre_trained=False' while creating the instance.")
            return 200
    
    def load_model(self):
        model_path = f'topic_models/{self.model_type}'
        if os.path.exists(model_path):
            self.model = BERTopic.load(model_path,embedding_model= self.embedding_model)
            return 100
        else:
            logging.error(f"> Model not found in the default path. Check whether saved model exists at {model_path}")
            return 200
        
    def get_topic_name(self,text):
        if self.model is not None:
            topics, _ = self.model.transform(text)
            cluster_id = topics[0]
            cluster_name = list(self.model.get_topic_info(cluster_id)['Name'])
            return (cluster_id,cluster_name)
        else:
            logging.error("> Model is not loaded. Make sure it is loaded or not.")

    def pre_process_text(self,documents):
        documents = [documents]
        clean_documents = []
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        for doc in documents:
            clean_sentences = ""
            sentences = sent_tokenize(doc)
            for sentence in sentences:
                clean_words = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if
                               word not in stop_words]
                cln_stn = " ".join(clean_words)
                clean_sentences += (cln_stn.strip())

            clean_documents.append(clean_sentences)

        return clean_documents[0]

    def get_possible_topics(self, term):
        if self.model is not None:
            return self.model.find_topics(term)
        else:
            logging.error("> Model is not loaded. Make sure it is loaded or not.")

    

