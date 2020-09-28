import numpy as np
import spacy
import glob


def build_dataset():

    feature_map = {}

    # spaCy english language model
    nlp = spacy.load("en_core_web_sm")

    # num_autos = 0
    # num_hockey = 0

    X = []
    y = []

    # read autos docs, ignoring header portions of files
    index = 0
    for file in glob.glob("../train/rec.autos/*"):
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
                    if token.text not in feature_map:
                        feature_map[token.text] = index
                        index += 1
                        curr_doc.append(1)
                    else:
                        curr_doc[feature_map[token.text]] += 1
            
            X.append(curr_doc)
            y.append(0)

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
                    if token.text not in feature_map:
                        feature_map[token.text] = index
                        index += 1
                        curr_doc.append(1)
                    else:
                        curr_doc[feature_map[token.text]] += 1
            
            X.append(curr_doc)
            y.append(1)

    print("Processed hockey training set!")

    # equalize the feature set lengths for all training examples
    for doc in X[:-1]:
        if len(doc) < len(X[-1]):
            doc.extend([0] * (len(X[-1]) - len(doc)))

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y, feature_map

if __name__ == "__main__":

    X, y, feature_map = build_dataset()
    