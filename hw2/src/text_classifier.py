from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import numpy as np
from preprocessing import build_dataset
import time

if __name__ == "__main__":

    # initialize multinomial NB with laplace smoothing and non-uniform, learned class probabilities from the data
    clf = MultinomialNB(alpha = 1.0, fit_prior = True)
    
    # load dataset after preprocessing

    print("Loading dataset...")
    start_load = time.time()
    X_train, y_train, X_test, y_test, feature_map = build_dataset()
    end = time.time()
    print("\nDataset loaded. Total time elapsed: {:.2f}s".format(end - start_load))

    print("\nNumber of training examples: ", X_train.shape[0])
    print("Number of test examples: ", X_test.shape[0])
    print("Number of unique words in feature map: ", X_train.shape[1])

    print("\nFitting Multinomial Naive Bayes model...")

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    
    print("\nTraining time elapsed: {:.2f}s".format(end - start))

    preds = clf.predict(X_test)
    pos_score = f1_score(y_test, preds, pos_label = 1, average = 'binary')
    neg_score = f1_score(y_test, preds, pos_label = 0, average = 'binary')
    acc = clf.score(X_test, y_test)

    #pred_acc = clf.score(X_test, y_test)
    print("\nF1 score for positive class (hockey): ", pos_score)
    print("F1 score for negative class (autos): ", neg_score)
    print("Mean Accuracy: {:.2f}%".format(acc * 100))

    end = time.time()
    print("\nTotal time elapsed: {:.2f}s".format(end - start_load))