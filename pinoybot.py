"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List
from pinoybot_trainModel import *
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    #    return tags

    # You can define other functions, import new libraries, or add other Python files as needed, as long as
    # the tag_language function is retained and correctly accomplishes the expected task.

    # Currently, the bot just tags every token as FIL. Replace this with your more intelligent predictions.

    # output list
    tags = []

    # load pickle file
    with open('trained_decisiontrees_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # initialzie previous word's language and previous word for certain features
    prev_language = -1
    prev_word = "."

    # loop each token, one-by-one extract features and predict
    for token in tokens:
        # extract features of current token being evaluated
        temp = featureExtraction(prev_language, token, prev_word)

        # convert to numpy array
        X_numpy = np.array(temp)

        # for reshaping because predict() doesn't accept a 1d array
        X_numpy = X_numpy.reshape(1, -1)
        
        # prediction
        prediction = model.predict(X_numpy)

        # clean up the prediction because the reshape has added unnecessary stuff to the array
        clean_label = prediction.item()
        
        # for "is Previous Word capitalized" feature
        prev_word = token

        # for "previous language" feature
        if (clean_label == "FIL"):
            prev_language = 0
        elif (clean_label == "ENG"):
            prev_language = 1
        elif (clean_label == "OTH"):
            prev_language = 2

        # append prediction to tags list
        tags.append(clean_label)
        
    return tags

if __name__ == "__main__":
    # Example usage
    # example_tokens = ["Mahal", "na", "mahal", "kita", ",", "ngunit", "I", "am", "still", "not", "ready", ".", "This", "life", "has", "been", "hirap", "for", "me", "and", "pinalunch", "pa", "me", "-"]
    
    input_tokens = ["Can", "you", "paki-abot", "the", "remote", "?", "I", "want", "to", "watch", "the", "news", "."]
    print("Tokens:", input_tokens)
    tags = tag_language(input_tokens)
    print(tags)
