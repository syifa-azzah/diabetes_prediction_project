from data.data_loader import load_data

def test_data_loading():

    df = load_data()

    assert df is not None

    print("✓ Data loading test passed")


if __name__ == "__main__":
    test_data_loading()
    