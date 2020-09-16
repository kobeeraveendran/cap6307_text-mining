
from sentiment_reader import SentimentCorpus
from multinomial_naive_bayes import MultinomialNaiveBayes
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Text classification with Naive Bayes")

    parser.add_argument("--smoothing", type = int, nargs = '?', dest = "smoothing", help = "Smoothing parameter. If <= 0, no smoothing is applied.")
    parser.add_argument("--train_split", type = float, nargs = '?', dest = "train_split", 
        help = "Training set percentage used to determine train/test split. For example, supplying --train_split=0.5 allocates half of the dataset for training and half for testing.")

    args = parser.parse_args()
    train_split = args.train_split
    smoothing = args.smoothing

    if not args.smoothing or args.smoothing <= 0:
        smoothing = 0

    if not args.train_split or args.train_split > 0.9 or args.train_split < 0.1:
        train_split = 0.8

    dataset = SentimentCorpus(train_per = train_split, test_per = round(1 - train_split, 2))
    
    if smoothing:
        nb = MultinomialNaiveBayes(smoothing = True, smoothing_param = smoothing)
    else:
        nb = MultinomialNaiveBayes(smoothing = False)
    
    params = nb.train(dataset.train_X, dataset.train_y)
    
    predict_train = nb.test(dataset.train_X, params)
    eval_train = nb.evaluate(predict_train, dataset.train_y)
    
    predict_test = nb.test(dataset.test_X, params)
    eval_test = nb.evaluate(predict_test, dataset.test_y)
    
    print("Accuracy on training set: %f, on test set: %f" % (eval_train, eval_test))
    print("Configuration:\n\tSmoothing: {}\n\tTrain/Test Split: {}/{}".format(smoothing, train_split*100, (round(1-train_split, 2))*100))



