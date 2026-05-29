from sklearn.metrics import accuracy_score

def test_accuracy_score():

    y_true = [1, 0, 1]

    y_pred = [1, 0, 1]

    accuracy = accuracy_score(y_true, y_pred)

    assert accuracy == 1.0

    print("✓ Evaluation test passed")


if __name__ == "__main__":
    test_accuracy_score()
    