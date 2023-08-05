import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from modules.eden_conX_engine.leaf_pipeline import CONX_LEAF_ML_Pipeline

conx_leaf_pipeline = CONX_LEAF_ML_Pipeline()


def make_response(map):
    return HttpResponse(content=json.dumps(map))

def check_field_validity(valid_fields, data):
    condition = True
    for field in valid_fields:
            if field not in data.keys():
                condition = False
    return condition

def throw_invalid_fields_error():
    response = {}
    response['messaage'] = "Valid fields not found in request body"
    response['status'] = 200
    return make_response(response)

def throw_http_method_not_supported_error():
     return HttpResponse(
            content=json.dumps({"status": 200, "message": "HTTP method is not supported."})
        )


@csrf_exempt
def use_leaf_text_pipeline(request):
    if request.method == "POST":
        data = json.loads(request.body)
        valid_fields = ['text_content', 'leaf_id']
        if check_field_validity(valid_fields,data):
            response = conx_leaf_pipeline.start_text_workflow(data['text_content'])
            return make_response(response)
        else:
           return throw_invalid_fields_error()
    else:
        return throw_http_method_not_supported_error()