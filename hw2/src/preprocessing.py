import numpy as np
import spacy
import glob

feature_dict = {}
feature_counts = {}

# spaCy english language model
nlp = spacy.load("en_core_web_sm")

# read autos docs, ignoring header portions of files
for file in glob.glob("../train/rec.autos/*"):
    with open(file) as f:
        all_lines = f.readlines()
        i = 0
        while "Lines:" not in all_lines[i]:
            i += 1

        num_lines = int(all_lines[i].split(":")[1])
        i += 1

        rel_string = ''.join([line[:-1] for line in all_lines[i:i + num_lines]])
        
        # extract parts of speech, stop word status, etc. for each token
        doc_tokens = nlp(rel_string)

        for token in doc_tokens:
            if token.pos_ != "X" and not token.is_stop:
                to_lower = token.text.lower()
                feature_counts[to_lower] = feature_dict.setdefault(to_lower, 0) + 1
