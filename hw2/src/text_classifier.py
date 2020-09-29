from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import numpy as np
from preprocessing import build_dataset

if __name__ == "__main__":

    # initialize multinomial NB with laplace smoothing and non-uniform, learned class probabilities from the data
    clf = MultinomialNB(alpha = 1.0, fit_prior = True)
    
    # load dataset after preprocessing
    X_train, y_train, X_test, y_test, feature_map = build_dataset()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    pos_score = f1_score(y_test, preds, pos_label = 1, average = 'binary')
    neg_score = f1_score(y_test, preds, pos_label = 0, average = 'binary')
    acc = clf.score(X_test, y_test)

    #pred_acc = clf.score(X_test, y_test)
    print("F1 score for positive class: ", pos_score)
    print("F1 score for negative class: ", neg_score)
    print("Accuracy: ", acc)

    print("\nPredictions:\n", preds)
    print("\nGround truth:\n", y_test)