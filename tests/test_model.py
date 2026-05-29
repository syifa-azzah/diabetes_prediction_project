import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_data
from preprocessing.preprocessing import clean_data
from models.model_training import get_models


def test_data_loading():

    df = load_data()

    assert df is not None

    print("✓ Data loading test passed")


def test_clean_data():

    df = load_data()

    df = clean_data(df)

    assert df.isnull().sum().sum() == 0

    print("✓ Data cleaning test passed")


def test_models_exist():

    models = get_models()

    assert len(models) > 0

    print("✓ Model initialization test passed")


if __name__ == "__main__":

    test_data_loading()

    test_clean_data()

    test_models_exist()

    print("\n✓ All tests passed successfully")