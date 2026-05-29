from data.data_loader import load_data
from preprocessing.preprocessing import clean_data

def test_clean_data():

    df = load_data()

    df = clean_data(df)

    assert df.isnull().sum().sum() == 0

    print("✓ Data preprocessing test passed")


if __name__ == "__main__":
    test_clean_data()
    