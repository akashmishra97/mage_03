if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd

@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    
    # Convert the DataFrame to a dictionary format suitable for DictVectorizer
    df = data
    train_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    
    # Fit the dict vectorizer
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    
    # Extract the target variable
    y_train = df['duration'].values
    
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print the intercept of the model
    print(f"Intercept: {model.intercept_}")
    
    # Return the dict vectorizer and the model
    return {
        'dict_vectorizer': dv,
        'model': model
    }

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert 'dict_vectorizer' in output, 'The output does not contain a dict vectorizer'
    assert 'model' in output, 'The output does not contain a model'
    assert isinstance(output['dict_vectorizer'], DictVectorizer), 'The dict vectorizer is not of type DictVectorizer'
    assert isinstance(output['model'], LinearRegression), 'The model is not of type LinearRegression'
