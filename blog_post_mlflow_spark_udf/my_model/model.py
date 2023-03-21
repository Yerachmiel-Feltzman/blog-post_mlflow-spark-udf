import logging
from packaging import version

import mlflow.pyfunc
import pandas as pd
from packaging.version import Version

logger = logging.getLogger(__name__)


# similar to documentation example: https://mlflow.org/docs/2.2.2/models.html#example-creating-a-custom-add-n-model
class MyModel(mlflow.pyfunc.PythonModel):
    NAME = "my_model"

    def __init__(self):
        # compatibility limitation
        self._min_compatible_pandas_version: Version = version.parse("1.1.0")
        self._max_compatible_pandas_version: Version = version.parse("1.2.5")

    def _check_compatibility(self):
        inference_pandas_version: Version = version.parse(pd.__version__)
        logger.info(f"running pandas version: {inference_pandas_version}")

        if inference_pandas_version < self._min_compatible_pandas_version \
                or inference_pandas_version > self._max_compatible_pandas_version:
            raise RuntimeError(
                "Running pandas version ('%s') is incompatible with min ('%s'} and max ('%s') versions."
                % (inference_pandas_version, self._min_compatible_pandas_version, self._max_compatible_pandas_version)
            )

    def predict(self, context, model_input: pd.DataFrame):
        self._check_compatibility()
        return model_input.apply(lambda column: column * 10)
