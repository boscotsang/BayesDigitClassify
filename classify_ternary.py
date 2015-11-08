import numpy
from sklearn.metrics import confusion_matrix

def load_data():
    train_labels = []
    with open('digitdata/traininglabels', 'rb') as f:
        for i, line in enumerate(f):
            train_labels.append(int(line))
    train_labels = numpy.array(train_labels, dtype=int)

    train_x = numpy.zeros((train_labels.shape[0] * 28 * 28))
    with open('digitdata/trainingimages', 'rb') as f:
        for i, line in enumerate(f):
            for j, char in enumerate(line.strip('\n')):
                if '+' == char:
                    train_x[i * 28 + j] = 1
                if '#' == char:
                    train_x[i * 28 + j] = 2

    train_x = numpy.array(train_x, dtype=int).reshape((train_labels.shape[0], 28 * 28))

    test_labels = []
    with open('digitdata/testlabels', 'rb') as f:
        for i, line in enumerate(f):
            test_labels.append(int(line))
    test_labels = numpy.array(test_labels, dtype=int)

    test_x = numpy.zeros((test_labels.shape[0] * 28 * 28))
    with open('digitdata/testimages', 'rb') as f:
        for i, line in enumerate(f):
            for j, char in enumerate(line.strip('\n')):
                if '+' == char:
                    test_x[i * 28 + j] = 1
                if '#' == char:
                    test_x[i * 28 + j] = 2

    test_x = numpy.array(test_x, dtype=int).reshape((test_labels.shape[0], 28 * 28))

    return train_x, train_labels, test_x, test_labels

class BayesClassifier(object):
    def __init__(self):
        self.bayesmatrix = None

    def fit(self, X, y):
        bayesmatrix = numpy.ones((10, 3, 28 * 28), dtype=numpy.float64)
        for k in xrange(10):
            for i in xrange(3):
                for j in xrange(X.shape[1]):
                    bayesmatrix[k, i, j] = numpy.sum(X[y==k, j]==i)
        numclass = numpy.zeros(10)
        for i in xrange(10):
            numclass[i] = numpy.sum(y==i) + 1
        bayesmatrix += 1
        bayesmatrix /= numclass[:, numpy.newaxis, numpy.newaxis]
        self.bayesmatrix = bayesmatrix

    def predict(self, X):
        labels = []
        for i in xrange(X.shape[0]):
            label = numpy.argmax(numpy.sum(numpy.log(self.bayesmatrix[:, 0, X[i, :]==0]), axis=1) +
                                numpy.sum(numpy.log(self.bayesmatrix[:, 1, X[i, :]==1]), axis=1) +
                                numpy.sum(numpy.log(self.bayesmatrix[:, 2, X[i, :]==2]), axis=1))
            labels.append(label)
        return numpy.array(labels)


if "__main__" == __name__:
    X, y, test_x, test_y = load_data()
    clf = BayesClassifier()
    clf.fit(X, y)
    pr = clf.predict(test_x)
    print "Confusion Matrix"
    print confusion_matrix(test_y, pr)
    print "Accuracy"
    print numpy.sum(pr == test_y) / float(test_y.shape[0])
