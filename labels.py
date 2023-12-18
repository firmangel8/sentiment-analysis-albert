
""" For class mode to predict labels assignment """
def binary_sentiment() -> dict[str, int]:
    """ For class mode to predict binary class of sentiment output  """
    return {'Positive': 0, 'Negative': 1}

def multiclass_sentiment() -> dict[str, int]:
    """ For class mode to predict multiclass of sentiment output  """
    return {'Very Positive': 0, 'Very Negative': 1, 'Mixed': 2, 'Positive': 3, 'Negative': 4, 'Neutral': 5}
