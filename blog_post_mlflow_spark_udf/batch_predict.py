import sys

import mlflow
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import struct

from blog_post_mlflow_spark_udf.my_model.model import MyModel

model_uri_template = f"models:/{MyModel.NAME}/%s"


def generate_dummy_input_data(spark: SparkSession) -> DataFrame:
    columns = ["id", "value"]
    data = [
        ("id_1", 1),
        ("id_2", 2),
        ("id_3", 3),
    ]
    return spark.createDataFrame(data).toDF(*columns)


def batch_predict(env_manager: str):
    spark = SparkSession \
        .builder \
        .getOrCreate()

    input_data = generate_dummy_input_data(spark)

    # for more on URIs format for loading models see:
    # https://www.mlflow.org/docs/2.2.2/concepts.html#referencing-artifacts
    model_uri = f"models:/{MyModel.NAME}/latest"

    predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, env_manager=env_manager)
    return input_data.withColumn("prediction", predict_udf(struct("value")))


if __name__ == '__main__':
    result = batch_predict(env_manager=(sys.argv[1]))
    result.show()