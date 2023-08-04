# Sample dictionary with string keys and integer values
d = {
    'a': 5,
    'b': 2,
    'c': 8,
    'd': 1
}


def sortBasedOnDict(dictionary: dict):
    """Takes a dictionary and returns the keys sorted from least to greatest
    based on their integer values.

    Args:
        dictionary (dict): Dictionary where the keys are strings and values are integers.

    Returns:
        list: list of the keys sorted based on their values from least to greatest.
    """
    # Sort the dictionary values from least to greatest
    itemsSorted = sorted(dictionary.items(), key=lambda item: item[1])

    keysSorted = []

    for item in itemsSorted:
        keysSorted.append(item[0])

    return keysSorted


print(sortBasedOnDict(d))
