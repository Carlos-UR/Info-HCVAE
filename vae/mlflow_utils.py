import os
import uuid

import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils import try_mlflow_log


class MLFLowLogger():

    def __init__(self, config):
        mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
        client = MlflowClient()
        run = client.create_run(config.get("mlflow_experiment_id"))
        self._run_id = run.info.run_id
        for key, value in config.items():
            try_mlflow_log(client.log_param, self._run_id, key, value)
        self.client = client

    def on_result(self, result):
        for key, value in result.items():
            if not isinstance(value, float) and not isinstance(value, np.float32):
                print("IGNORING", key, value, type(value))
                continue
            try_mlflow_log(self.client.log_metric,
                self._run_id, key, value, step=result["epoch"])

    def on_checkpoint(self, checkpoint_path):
        #os.makedirs(checkpoint_path, exist_ok=True)
        try_mlflow_log(self.client.log_artifact, self._run_id, checkpoint_path)

    def close(self):
        self.client.set_terminated(self._run_id)