import logging
from base64 import b64decode
from datetime import datetime

from flask import request
from flask_restful import Resource
from pydantic.error_wrappers import ValidationError

from settings import Settings
from v1.routines.iris_classification import IrisClassification
from v1.schemas.payloads import (
    ClassifyPayload,
    ModelResponse,
    ErrorDescription,
    Response,
)


class ApiClassify(Resource):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cfg = Settings()
        self.iris_classification = IrisClassification(self.cfg, self.cfg.model)

    def get(self):
        return {"success": "ok"}, 200

    def post(self):
        json_data = request.get_json(force=True)

        self.logger.info(f"Processing started at {str(datetime.now())}")
        try:
            if "base64" in json_data:
                payload = b64decode(json_data["base64"])
            else:
                payload = json_data

            ClassifyPayload.parse_obj(json_data)
            pred_confidence, pred_label = self.iris_classification.main_routine(payload)

        except (ValidationError, Exception) as e:
            error_description = ErrorDescription(
                raised=type(e).__name__,
                raisedOn="ApiClassify",
                message=str(e),
                code="400",
            )
            response = Response(
                message="failed",
                data=None,
                error=error_description,
                version=self.cfg.version,
            )
            return response.dict(), 400

        response = Response(
            message="success",
            data=ModelResponse(
                pred_confidence=pred_confidence,
                pred_label=pred_label,
                model_version=self.cfg.model._model_meta.run_id,
            ),
            error=None,
            version=self.cfg.version,
        )
        self.logger.info(f"Request processed with success at {str(datetime.now())}")
        return response.dict(), 200
