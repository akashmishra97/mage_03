import requests
from io import BytesIO
from typing import List
import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    # Iterate only for March (month 3)
    for year, month in [(2023, 3)]:
        url = (
            'https://github.com/akashmishra97/mlops-zoomcamp/raw/main/data'
            f'/{year}/{month:02d}.parquet'
        )
        print(f"Fetching URL: {url}")

        response = requests.get(url)
        print(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: {response.text}")

        print(f"Content Type: {response.headers['Content-Type']}")
        print(f"First 100 bytes of content: {response.content[:100]}")

        df = pd.read_parquet(BytesIO(response.content))
        dfs.append(df)

    return pd.concat(dfs)
