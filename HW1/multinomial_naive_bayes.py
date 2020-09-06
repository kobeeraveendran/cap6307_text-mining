import numpy as np
from linear_classifier import LinearClassifier
np.set_printoptions(threshold = np.inf)

class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = True
        self.smooth_param = 1
        
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

        #bags = [[0] * n_words] * n_classes
        bag_sizes = [0] * n_classes

        for doc in range(len(x)):
            c = y[doc][0]

            for word in range(len(x[doc])):
                likelihood[word, c] += x[doc][word]
                # bags[c][word] += x[doc][word]
                bag_sizes[c] += 1

        # for doc in range(len(x)):
            
        #     c = y[doc][0]
        #     for word in x[doc]:
        #         bags[c][word] = bags[c].setdefault(int(word), 0) + 1
        #         bag_sizes[c] += 1


        # print(bags[0])
        # print(bag_sizes)

        for i in range(len(likelihood)):
            for j in range(len(likelihood[0])):
                likelihood[i][j] += 1
                likelihood[i][j] /= (bag_sizes[j] + n_words)
        
        # for c in range(len(bags)):
        #     for word, count in bags[c].items():
        #         likelihood[word, c] = (count + 1) / (bag_sizes[c] + n_words)


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
