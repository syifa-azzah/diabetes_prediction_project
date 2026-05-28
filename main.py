from data.data_loader import load_data
from preprocessing.preprocessing import clean_data
from preprocessing.preprocessing import split_data
from preprocessing.preprocessing import scale_data

from models.model_training import get_models

from evaluation.evaluation import evaluate_models

from visualization.visualization import plot_histograms
from visualization.visualization import plot_heatmap


def main():

    print("Loading dataset...")

    df = load_data()

    print("Cleaning dataset...")

    df = clean_data(df)

    print("Creating visualizations...")

    plot_histograms(df)
    plot_heatmap(df)

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = split_data(df)

    print("Scaling dataset...")

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    print("Training models...")

    models = get_models()

    results = evaluate_models(
        models,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    print("\nMODEL RESULTS:\n")

    for model_name, scores in results.items():

        print(model_name)

        print("Train Accuracy:", scores["Train Accuracy"])
        print("Test Accuracy:", scores["Test Accuracy"])

        print()


if __name__ == "__main__":
    main()