if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def train_model(data, *args, **kwargs):
    """
    Train a linear regression model using the pickup and dropoff location IDs.

    Args:
        data (pandas.DataFrame): The input data.

    Returns:
        Tuple[DictVectorizer, LinearRegression]: The fitted dictionary vectorizer and the trained linear regression model.
    """
    import pandas as pd
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LinearRegression

    # Convert location IDs to strings
    categorical = ['PULocationID', 'DOLocationID']
    data[categorical] = data[categorical].astype(str)

    # Extract the target variable (duration)
    target = data['duration']

    # Extract the features (pickup and dropoff location IDs)
    features = data[categorical].to_dict(orient='records')

    # Fit the dictionary vectorizer
    dv = DictVectorizer()
    X = dv.fit_transform(features)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, target)

    # Print the model intercept
    print(f"Model intercept: {model.intercept_}")

    return dv, model

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
