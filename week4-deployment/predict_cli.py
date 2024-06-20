import pickle
import sklearn
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import argparse

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


def prepare_features(ride):
    features = {}
    features["PU_DO"] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features["trip_distance"] = ride["trip_distance"]
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)

    return preds

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def main():

    parser = argparse.ArgumentParser(description="Predict trip durations.")
    parser.add_argument('--year', type=int, default=2023, help='Year of the trip data (default: 2023)')
    parser.add_argument('--month', type=int, default=5, help='Month of the trip data (default: 4)')
    
    args = parser.parse_args()

    year = args.year
    month = args.month

    if (year == 2023) and (month == 4):
        data_path = 'data/yellow_tripdata_2023-04.parquet'
    elif (year == 2023) and (month == 5):
        data_path = 'data/yellow_tripdata_2023-05.parquet'
    elif (year == 2023) and (month == 3):
        data_path = 'data/yellow_tripdata_2023-03.parquet'
    else:
        print("Not implemented")

    df = read_data(data_path)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"{np.mean(y_pred)=}")

if __name__ == "__main__":
    main()
