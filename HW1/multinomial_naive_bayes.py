import numpy as np
from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self, smoothing = True, smoothing_param = 1):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = smoothing
        self.smooth_param = smoothing_param
        
    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words 
        n_docs, n_words = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        
        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words,n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
            # prior[0] is the prior probability of a document being of class 0
            # likelihood[4, 0] is the likelihood of the fifth(*) feature being 
            # active, given that the document is of class 0
            # (*) recall that Python starts indices at 0, so an index of 4 
            # corresponds to the fifth feature!
        
        ###########################


        # YOUR CODE HERE

        # count the number of occurrences of each class in training set, convert into probability
        prior = np.array([np.count_nonzero(y == [i]) / len(y) for i in classes])

        # keep track of total number of words belonging to class c
        bag_sizes = [0] * n_classes

        # count category-wise occurrences of words, track num. of words per category
        for doc in range(len(x)):
            c = y[doc][0]

            for word in range(len(x[doc])):
                likelihood[word, c] += x[doc][word]
                bag_sizes[c] += 1

        # convert word frequencies into probabilities (with or without smoothing)
        for word in range(len(likelihood)):
            for c in range(len(likelihood[0])):
                if self.smooth:
                    likelihood[word, c] = (likelihood[word, c] + self.smooth_param) / (bag_sizes[c] + n_words)
                else:
                    likelihood[word, c] /= bag_sizes[c]

        ###########################

        params = np.zeros((n_words+1,n_classes))
        for i in range(n_classes): 
            # log probabilities
            params[0,i] = np.log(prior[i])
            with np.errstate(divide='ignore'): # ignore warnings
                params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
