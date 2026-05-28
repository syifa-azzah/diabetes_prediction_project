import pandas as pd

def load_data():

    df = pd.read_csv("data/diabetes.csv")

    return df