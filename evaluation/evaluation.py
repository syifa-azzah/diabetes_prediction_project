from sklearn.metrics import accuracy_score

def evaluate_models(models, X_train, X_test, y_train, y_test):

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        results[name] = {
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy
        }

    return results