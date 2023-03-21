import logging
from pathlib import Path

import mlflow
from mlflow.models.model import ModelInfo

from blog_post_mlflow_spark_udf.my_model.model import MyModel

logger = logging.getLogger(__name__)


def register():
    my_model = MyModel()

    model_info: ModelInfo = mlflow.pyfunc.log_model(artifact_path=my_model.NAME,
                                                    registered_model_name=my_model.NAME,
                                                    python_model=my_model,
                                                    code_path=[str(Path("blog_post_mlflow_spark_udf"))],
                                                    conda_env=str(Path("blog_post_mlflow_spark_udf", "my_model",
                                                                       "model-env.yml")))
    return model_info


if __name__ == '__main__':
    register()
