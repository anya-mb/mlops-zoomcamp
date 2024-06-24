import pickle
import numpy as np
import pandas as pd
import argparse
import requests
from io import BytesIO


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

def read_data(url):
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses

    file_content = BytesIO(response.content)

    df = pd.read_parquet(file_content)
    
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

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"{np.mean(y_pred)=}")

if __name__ == "__main__":
    main()
