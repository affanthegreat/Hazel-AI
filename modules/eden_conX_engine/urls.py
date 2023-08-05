from django.urls import path
from modules.eden_conX_engine.views import *

urlpatterns = [
    path('leaf_text_pipeline', use_leaf_text_pipeline),
    path('comment_text_pipeline',use_comment_text_pipeline)
]