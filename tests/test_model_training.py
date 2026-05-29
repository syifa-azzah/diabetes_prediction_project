from models.model_training import get_models

def test_models_exist():

    models = get_models()

    assert len(models) > 0

    print("✓ Model initialization test passed")


if __name__ == "__main__":
    test_models_exist()
    