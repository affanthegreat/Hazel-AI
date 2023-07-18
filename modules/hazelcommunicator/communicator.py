import logging
import requests


class HazelComunicator():
    def __init__(self,hazel_core_server):
        logging.info("Hazel Communicator Object has been created.")

        self.HAZEL_SERVER = hazel_core_server
        self.API_SUB_URL = self.HAZEL_SERVER + 'communicator/'
        create_url = lambda path: self.API_SUB_URL + path


        self.STREAM_USERS_BY_TOPIC_ENDPOINT = create_url('stream_users_by_topic')
        self.UPDATE_USER_TOPICS_ENDPOINT = create_url('update_user_topics')

        # Leaf Side Endpoints
        self.STREAM_UNCATEGORIZED_LEAVES_ENDPOINT = create_url('stream_uncategorized_leaves')
        self.STREAM_LEAVES_TOPIC_WISE_ENDPOINT = create_url('stream_leaves_topic_wise')
        self.STREAM_NEGATIVE_LEAVES_ENDPOINT = create_url('stream_negative_leaves')
        self.STREAM_UNMARKED_COMMENTS_ENDPOINT = create_url('stream_unmarked_comments')
        self.STREAM_MARKED_COMMENTS_ENDPOINT = create_url('stream_marked_comments')
        self.SEND_LEAF_METRICS_ENDPOINT = create_url('send_leaf_metrics')
        self.UPDATE_BATCH_LEAF_METRICS = create_url('update_batch_leaf_metrics')

    def make_post_request(self,api_endpoint,body):
        return requests.post(api_endpoint,json=body)

    def stream_users_by_topic(self, topic_id,page_number=1):
        request_body = {'page_number': page_number, 'topic_id': topic_id}
        response= self.make_post_request(self.STREAM_USERS_BY_TOPIC_ENDPOINT,  request_body)
        return response

    def update_user_metrics(self, topic_id, user_id):
        request_body = {'user_id': user_id, 'topic_id': topic_id}
        response = self.make_post_request(self.UPDATE_USER_TOPICS_ENDPOINT, request_body)
        return response

    def get_leaf_metrics(self, leaf_id):
        request_body = {'leaf_id': leaf_id}
        response = self.make_post_request(self.SEND_LEAF_METRICS_ENDPOINT, request_body)
        return response

    def stream_uncategorized_leaves(self,page_number=1):
        request_body = {'page_number': page_number}
        response = self.make_post_request(self.STREAM_UNCATEGORIZED_LEAVES_ENDPOINT,request_body)
        return response

    def stream_leaves_topic_wise(self,topic_id, page_number= 1):
        request_body = {'page_number': page_number, 'topic_id': topic_id}
        response = self.make_post_request(self.STREAM_LEAVES_TOPIC_WISE_ENDPOINT, request_body)
        return response

    def stream_negative_leaves(self,page_number=1):
        request_body = {'page_number': page_number}
        response = self.make_post_request(self.STREAM_NEGATIVE_LEAVES_ENDPOINT,request_body)
        return response

    def stream_unmarked_comments(self,leaf_id, page_number=1):
        request_body = {'page_number': page_number, 'leaf_id':leaf_id}
        response = self.make_post_request(self.STREAM_UNMARKED_COMMENTS_ENDPOINT,request_body)
        return response

    def stream_marked_comments(self,leaf_id, page_number=1):
        request_body = {'page_number': page_number, 'leaf_id':leaf_id}
        response = self.make_post_request(self.STREAM_MARKED_COMMENTS_ENDPOINT,request_body)
        return response

    def update_batch_leaf_metrics(self,batch_number,leaves_collection):
        request_body = {'batch_number':batch_number, 'leaves_collection': leaves_collection}
        response = self.make_post_request(self.UPDATE_BATCH_LEAF_METRICS, request_body)
        return response
