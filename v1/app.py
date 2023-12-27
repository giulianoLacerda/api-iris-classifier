from flask import Blueprint
from flask_restful import Api

from v1.services.api_classify import ApiClassify

api_bp = Blueprint("v1", __name__)
api = Api(api_bp)

api.add_resource(ApiClassify, "/classify")
