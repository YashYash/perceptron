import pandas as pd
from models import Perceptron


def run():
    df = pd.read_csv(r'/Users/yashsaxena/Desktop/udacity/perceptron/data.csv')
    np_data = df.to_numpy()

    perceptron = Perceptron(
        data=np_data
    )

    perceptron.train()
