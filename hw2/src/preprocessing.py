import numpy as np
import spacy
import glob
from sklearn.utils import shuffle

def build_dataset():

    feature_map = {}

    # spaCy english language model
    nlp = spacy.load("en_core_web_sm")

    # num_autos = 0
    # num_hockey = 0

    X_train = []
    y_train = []

    # read autos docs, ignoring header portions of files
    index = 0
    for filenum, file in enumerate(glob.glob("../train/rec.autos/*")):
        #print(file)
        with open(file, encoding = 'utf8', errors = 'ignore') as f:
            all_lines = f.readlines()
            i = 0
            while i < len(all_lines) and "Lines:" not in all_lines[i]:
                i += 1

            if i >= len(all_lines):
                i = 5
                num_lines = len(all_lines) - i
            else:
                num_lines = int(all_lines[i].split(":")[1])
                i += 1

            rel_string = ''.join([line[:-1] for line in all_lines[i:i + num_lines]])
            
            # extract parts of speech, stop word status, etc. for each token
            doc_tokens = nlp(rel_string)

            # track word frequencies within the current document
            # feature_counts = {}
            curr_doc = [0] * len(feature_map)

            for token in doc_tokens:
                if token.pos_ != 'X' and token.text.isalpha() and not token.is_stop:
                    token_lower = token.text.lower()
                    # feature_counts[token_lower] = feature_counts.setdefault(token_lower, 0) + 1
                    if token_lower not in feature_map:
                        feature_map[token_lower] = index
                        index += 1
                        curr_doc.append(1)
                    else:
                        curr_doc[feature_map[token_lower]] += 1
                    
                    # if filenum == 0 :
                    #     print("Valid token: ", token_lower, "  Index = ", feature_map[token_lower])
            
            X_train.append(curr_doc)
            y_train.append(0)

    print("Processed autos training set!")

    # read hockey docs, ignoring headers
    for file in glob.glob("../train/rec.sport.hockey/*"):

        with open(file, encoding = 'utf8', errors = 'ignore') as f:

            all_lines = f.readlines()
            i = 0
            while i < len(all_lines) and "Lines:" not in all_lines[i]:
                i += 1
            
            if i >= len(all_lines):
                i = 5
                num_lines = len(all_lines) - i
            else:
                num_lines = int(all_lines[i].split(":")[1])
                i += 1

            rel_string = ''.join([line[:-1] for line in all_lines[i:i + num_lines]])
            
            # extract parts of speech, stop word status, etc. for each token
            doc_tokens = nlp(rel_string)

            # track word frequencies within the current document
            # feature_counts = {}
            curr_doc = [0] * len(feature_map)

            for token in doc_tokens:
                if token.pos_ != 'X' and token.text.isalpha() and not token.is_stop:
                    token_lower = token.text.lower()
                    # feature_counts[token_lower] = feature_counts.setdefault(token_lower, 0) + 1
                    if token_lower not in feature_map:
                        feature_map[token_lower] = index
                        index += 1
                        curr_doc.append(1)
                    else:
                        curr_doc[feature_map[token_lower]] += 1
            
            X_train.append(curr_doc)
            y_train.append(1)

    print("Processed hockey training set!")

    # equalize the feature set lengths for all training examples
    for doc in X_train[:-1]:
        if len(doc) < len(X_train[-1]):
            doc.extend([0] * (len(X_train[-1]) - len(doc)))

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    X_train, y_train = shuffle(X_train, y_train)

    X_test = []
    y_test = []

    # parse and load in test set
    for file in glob.glob("../test/rec.autos/*"):

        with open(file, encoding = 'utf8', errors = 'ignore') as f:

            all_lines = f.readlines()
            i = 0
            while i < len(all_lines) and "Lines:" not in all_lines[i]:
                i += 1
            
            if i >= len(all_lines):
                i = 5
                num_lines = len(all_lines) - i
            else:
                num_lines = int(all_lines[i].split(":")[1])
                i += 1

            rel_string = ''.join([line[:-1] for line in all_lines[i:i + num_lines]])
            
            # extract parts of speech, stop word status, etc. for each token
            doc_tokens = nlp(rel_string)

            # track word frequencies within the current document
            # feature_counts = {}
            curr_doc = [0] * len(feature_map)

            for token in doc_tokens:
                if token.pos_ != 'X' and token.text.isalpha() and not token.is_stop:
                    
                    token_lower = token.text.lower()
                    if token_lower in feature_map:
                        curr_doc[feature_map[token_lower]] += 1
            
            X_test.append(curr_doc)
            y_test.append(0)

    print("Processed autos test set!")
    
    for file in glob.glob("../test/rec.sport.hockey/*"):

        with open(file, encoding = 'utf8', errors = 'ignore') as f:

            all_lines = f.readlines()
            i = 0
            while i < len(all_lines) and "Lines:" not in all_lines[i]:
                i += 1
            
            if i >= len(all_lines):
                i = 5
                num_lines = len(all_lines) - i
            else:
                num_lines = int(all_lines[i].split(":")[1])
                i += 1

            rel_string = ''.join([line[:-1] for line in all_lines[i:i + num_lines]])
            
            # extract parts of speech, stop word status, etc. for each token
            doc_tokens = nlp(rel_string)

            # track word frequencies within the current document
            # feature_counts = {}
            curr_doc = [0] * len(feature_map)

            for token in doc_tokens:
                if token.pos_ != 'X' and token.text.isalpha() and not token.is_stop:
                    token_lower = token.text.lower()
                    # feature_counts[token_lower] = feature_counts.setdefault(token_lower, 0) + 1
                    if token_lower in feature_map:
                        curr_doc[feature_map[token_lower]] += 1
            
            X_test.append(curr_doc)
            y_test.append(1)

    print("Processed hockey test set!")

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    X_test, y_test = shuffle(X_test, y_test)

    return X_train, y_train, X_test, y_test, feature_map

if __name__ == "__main__":

    X_train, y_train, X_test, y_test, feature_map = build_dataset()
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)
    print("Feature set size: ", len(feature_map))