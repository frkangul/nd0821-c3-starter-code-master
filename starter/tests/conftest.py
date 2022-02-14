import pytest
import pandas as pd


@pytest.fixture(scope="session")
def data():
    file_path = "/starter/data/census_cleaned.csv"
    sample = pd.read_csv(file_path)
    return sample
