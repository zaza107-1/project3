from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.dataset import Dataset



url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv'
###dataset = TabularDatasetFactory.from_delimited_files(path = url)
###ds = dataset.to_pandas_dataframe() ### YOUR CODE HERE ###
dataset=Dataset.Tabular.from_delimited_files(path=url)



# TODO: Split data into train and test sets.

### YOUR CODE HERE ###a
run = Run.get_context()

def clean_data(data):
    
    # Clean and separate
    x_df = data.to_pandas_dataframe().dropna()
    y_df = x_df.pop("DEATH_EVENT")
    return x_df , y_df

x, y = clean_data(dataset)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)




def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model_logreg.joblib')



if __name__ == '__main__':
    main()