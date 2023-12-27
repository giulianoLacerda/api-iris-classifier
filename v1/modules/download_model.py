import mlflow
import os
import logging
from settings import Settings, BASE_DIR
from cloudpickle import dump

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    cfg = Settings()
    cfg.set_env(os.environ["MODE_DEPLOY"])
    if not os.path.exists(os.path.join(f"{BASE_DIR}/v1/models", "model.pkl")):
        logger.info("Downloading model from MlFlow...")
        mlflow.set_tracking_uri(cfg.mlflow_uri)
        mlflow.set_experiment(experiment_id="1")
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{cfg.model_name}/{cfg.model_stage}"
        )
        with open(os.path.join(BASE_DIR, "v1/models/model.pkl"), "wb") as pkl_file:
            dump(model, pkl_file)
    logger.info("Model downloaded with success!")
