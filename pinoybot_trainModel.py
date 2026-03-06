import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import List
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

vowels = ["a", "e", "i", "o", "u"]

eng_letters = ["c", "f", "j", "q", "v", "w", "x", "y", "z"]
fil_specWords = ["ay", "ng", "sa", "si", "ni", "na", "pa", "ko", "mo", "po", "ka", "ba", "ang", "mga", "oo"]
fil_prefixes = ["maki", "um", "paki", "nag", "pina", "naka", "pag", "mag", "i", "ka", "tag", "ma", "na"]
fil_suffixes = ["an", "in", "ng", "han", "hin", "daan"]
eng_prefixes = ["un", "re", "in", "im", "pre", "dis", "sub", "non", "mis", "over", "anti"]
eng_suffixes = ["tion", "ment", "ly", "able", "ness", "ful", "ed", "ing", "s", "ism", "ist", "ize", "ous", "ic"]

fil_bigrams = ["ng", "an", "in", "ka", "sa", "ta", "to", "ti", "da", "di"]
fil_trigrams = ["nan", "nga", "ang", "isa", "iya", "min"]
eng_bigrams = ["th", "er", "st", "es", "ou", "sh", "ee", "ll", "al"]
eng_trigrams = ["ion", "the", "ght", "all", "eak", "str", "chr"]

# returns true if token is all capitalized
def isAllCapitalized(token):
    return token.isupper()

# returns true if token is capitalized but not the first word
def isCapitalized(token, prev_token):
    if (not(prev_token == "." or prev_token == "!" or prev_token == "?")):
        if (token!= ""):
            if (token[0].isupper()):
                return True
    return False

# returns true if token is all alphabetical
def isAlphabet(token): 
    for i in range(len(token)):
        if (not(token[i].isalpha())):
            return False
    return True

# returns true if token has low vowel density
def getVowelDensity(token):
    vowel_count = 0
    if (token != ""):
        for i in token:
            if (i in vowels):
                vowel_count += 1
        return vowel_count / len(token)

    # if token is blank
    return 0

# returns true if token contains numerics
def containsNumerics(token):
    for i in range(len(token)):
        if (token[i].isdigit()):
            return True
    return False
    
# returns true if token contains symbols
def containsSymbols(token):
    for i in range(len(token)):
        if (not(token[i].isalnum())):
            return True
    return False

# returns true if token contains repeating characters (appearing >= 3 times)
def containsRepeatingChars(token):
    if (len(token) < 3):
        return False
    for i in range(len(token) - 2):
        if (token[i] == token[i + 1] and token[i] == token[i + 2]):
            return True
    return False

# returns true if token contains English letters
def containsEngLetters(token):
    for i in range(len(token)):
        if (token[i] in eng_letters):
            return True
    return False

# returns true if token is a Filipino functional word
def isFilSpecWord(token):
    return token in fil_specWords

# (helper function) returns NGram density of the token 
def calculateNGramDensity(token, ngrams):
    if (token != ""):
        token = token.lower()
        length = len(ngrams[0])
        if (len(token) < length):
            return 0
        count = 0
        ngrams_set = set(ngrams)
        for i in range(len(token) - length + 1):
            if (token[i:i+length] in ngrams_set):
                count += 1
        return (count / (len(token) - length + 1))
    
    # if token is blank
    return 0

# returns Filipino Bigram density of the token
def getFilBigramDensity(token):
    return calculateNGramDensity(token, fil_bigrams)

# returns Filipino Trigram density of the token
def getFilTrigramDensity(token):
    return calculateNGramDensity(token, fil_trigrams)

# returns English Bigram density of the token
def getEngBigramDensity(token):
    return calculateNGramDensity(token, eng_bigrams)

# returns English Trigram density of the token
def getEngTrigramDensity(token):
    return calculateNGramDensity(token, eng_trigrams)

# returns true if token contains Filipino prefix
def containsFilPrefix(token):
    token = token.lower()
    for i in fil_prefixes:
        if (token.startswith(i) and len(token) > len(i)):
            return True
    return False

# returns true if token contains Filipino suffix
def containsFilSuffix(token):
    token = token.lower()
    for i in fil_suffixes:
        if (token.endswith(i) and len(token) > len(i)):
            return True
    return False

# returns true if token contains English prefix
def containsEngPrefix(token):
    token = token.lower()
    for i in eng_prefixes:
        if (token.startswith(i) and len(token) > len(i)):
            return True
    return False

# returns true if token contains English suffix
def containsEngSuffix(token):
    token = token.lower()
    for i in eng_suffixes:
        if (token.endswith(i) and len(token) > len(i)):
            return True
    return False

# returns true if token contains Filipino reduplication (words like araw-araw)
def containsFilRedup(token):
    token = token.lower()
    if '-' in token:
        word = token.split('-')
        if len(word) == 2:
            if (word[0] == word[1]):
                return True
    return False

# returns the length of the token
def getWordLength(token):
    return len(token)

# feature extraction for input tokens
def featureExtraction(prev_language, token, prev_word):

    temp = []

    # extract feature using functions, then append to array
    temp.append(isAllCapitalized(token))
    temp.append(isCapitalized(token, prev_word))
    temp.append(isAlphabet(token))
    temp.append(getVowelDensity(token))
    temp.append(containsNumerics(token))
    temp.append(containsSymbols(token))
    temp.append(containsRepeatingChars(token))
    temp.append(containsEngLetters(token))
    temp.append(isFilSpecWord(token))
    temp.append(getFilBigramDensity(token))
    temp.append(getFilTrigramDensity(token))
    temp.append(getEngBigramDensity(token))
    temp.append(getEngTrigramDensity(token))
    temp.append(containsFilPrefix(token))
    temp.append(containsFilSuffix(token))
    temp.append(containsEngPrefix(token))
    temp.append(containsEngSuffix(token))
    temp.append(containsFilRedup(token))
    temp.append(getWordLength(token))
    temp.append(prev_language)

    return temp

if __name__ == '__main__':
    
    label = []
    featureMatrix = []
    temp = []

    # file read aggregated dataset
    with open('final_annotations_for_read.csv', mode='r') as file:

        # skip header line
        header_line = next(file)
        csv_reader = csv.reader(file)
        
        # initialize prev word and prev language
        prev_word = "."
        prev_language = -1

        # loop through dataset
        for row in csv_reader:
            
            # feature extraction of the token currently being read
            temp.append(isAllCapitalized(row[2]))
            temp.append(isCapitalized(row[2], prev_word))
            temp.append(isAlphabet(row[2]))
            temp.append(getVowelDensity(row[2]))
            temp.append(containsNumerics(row[2]))
            temp.append(containsSymbols(row[2]))
            temp.append(containsRepeatingChars(row[2]))
            temp.append(containsEngLetters(row[2]))
            temp.append(isFilSpecWord(row[2]))
            temp.append(getFilBigramDensity(row[2]))
            temp.append(getFilTrigramDensity(row[2]))
            temp.append(getEngBigramDensity(row[2]))
            temp.append(getEngTrigramDensity(row[2]))
            temp.append(containsFilPrefix(row[2]))
            temp.append(containsFilSuffix(row[2]))
            temp.append(containsEngPrefix(row[2]))
            temp.append(containsEngSuffix(row[2]))
            temp.append(containsFilRedup(row[2]))
            temp.append(getWordLength(row[2]))
            temp.append(prev_language)

            # assign previous token to currently-evaluating token
            prev_word = row[2]
            
            # put the features in featureMatrix
            featureMatrix.append(temp)

            # append tag of token to labels
            label.append(row[3])

            # for previous language feature
            if (row[3] == "FIL"):
                prev_language = 0
            elif (row[3] == "ENG"):
                prev_language = 1
            elif (row[3] == "OTH"):
                prev_language = 2

            # reset temp array
            temp = []

        # convert necessary arrays to numpy arrays for fit()
        X_numpy = np.array(featureMatrix)
        y_numpy = np.array(label)

    # split dataset to train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy, test_size=0.3)
    # split dataset to test set and validation set
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5)
    
    model = DecisionTreeClassifier(max_depth=20)
    model.fit(X_train, y_train)

#    plotting functions
#    plt.figure(figsize=(16,10), dpi = 70)
#    plot_tree(model)
#    plt.show()
    
    predictions = model.predict(X_test)
    predictions2 = model.predict(X_valid)

    # test set
    # show precision, recall, accuracy, and F1-Score
    report = classification_report(y_test, predictions)
    print("\nClassification Report for X_test:")
    print(report)

    #validation set
    # show precision, recall, accuracy, and F1-Score
    report2 = classification_report(y_valid, predictions2)
    print("\nClassification Report for X_valid:")
    print(report2)
    
    # save model to pickle file
    with open('trained_decisiontrees_model.pkl', 'wb') as file:
        pickle.dump(model, file)
        print("File saved successfully")

        
