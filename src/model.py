import os
import pandas as pd


def import_iris_data():
    dirname = os.path.dirname(__file__)
    iris_data_path = os.path.join(dirname, "./SSI_data_preprocessed.csv")
    iris_data = pd.read_csv(
        iris_data_path,
        header=None,
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
    )
    return iris_data