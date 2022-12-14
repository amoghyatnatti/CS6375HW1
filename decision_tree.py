# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
# Tahmid Imran and Amogh Yatnatti
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
from sklearn.metrics import confusion_matrix
import random
from sklearn import preprocessing


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    v = {}
    for i in range(len(x)):
        if x[i] not in v:
            v[x[i]] = []
        v[x[i]].append(i)
    return v


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    part = partition(y)
    # print(len(part[1])/1)
    totalCount = len(y)

    Hz = 0.0
    for key in part:
        vCount = len(part.get(key))
        vProb = vCount/totalCount
        Hz += -vProb * math.log(vProb, 2)
    # print(Hz)
    return Hz


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    Hy = entropy(y)
    xVal, xCounts = np.unique(x, return_counts=True)

    Hyx = 0.0
    for i in range(len(xVal)):
        prob = xCounts[i]/len(x)
        Hyx += prob*entropy(y[x == xVal[i]])

    return Hy-Hyx


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    if len(set(y)) == 1:
        return y[0]

    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for i in range(len(x[0])):
            uniqueVals = np.unique(x[:, i])
            for j in range(len((uniqueVals))):
                attribute_value_pairs.append((i, uniqueVals[j]))

    if len(attribute_value_pairs) == 0 or depth == max_depth:
        # Finds all unique elements and their positions
        unique, pos = np.unique(y, return_inverse=True)
        counts = np.bincount(pos)  # Count the number of each unique element
        maxpos = counts.argmax()
        return unique[maxpos]

    node = {}

    maxInfo = 0
    maxVariable = None
    for (k, v) in attribute_value_pairs:
        info = mutual_information(np.array(x[:, k] == v), y)
        if info > maxInfo:
            maxInfo = info
            maxVariable = (k, v)

    xTrue = x[(x[:, maxVariable[0]] == maxVariable[1])]
    xFalse = x[(x[:, maxVariable[0]] != maxVariable[1])]
    yTrue = y[(x[:, maxVariable[0]] == maxVariable[1])]
    yFalse = y[(x[:, maxVariable[0]] != maxVariable[1])]
    node[(maxVariable[0], maxVariable[1], False)] = id3(xFalse, yFalse,
                                                        attribute_value_pairs=attribute_value_pairs, depth=depth+1, max_depth=max_depth)
    node[(maxVariable[0], maxVariable[1], True)] = id3(xTrue, yTrue,
                                                       attribute_value_pairs=attribute_value_pairs, depth=depth+1, max_depth=max_depth)

    return node


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    if isinstance(tree, (int, np.int64, np.int32, np.str_)):
        return tree
    for key in tree.keys():
        present = x[key[0]] == key[1]
        predicted_label = predict_example(x, tree[(key[0], key[1], present)])
        break
    return predicted_label


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    count = 0
    for i in range(len(y_true)):
        count += 0 if y_true[i] == y_pred[i] else 1
    return count / len(y_true)


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print(
            '+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def learning_curve(Xtrn, ytrn, Xtst, ytst, title):
    training_errors = []
    testing_errors = []
    for i in range(1, 11):
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        y_pred_train = [predict_example(x, decision_tree) for x in Xtrn]
        training_errors.append(compute_error(ytrn, y_pred_train))
        y_pred_test = [predict_example(x, decision_tree) for x in Xtst]
        testing_errors.append(compute_error(ytst, y_pred_test))
        if i == 10:
            visualize(decision_tree)

    print(f'Training Error: {training_errors[:-1]}')
    print(f'Testing Error: {testing_errors[:-1]}')
    depths = np.arange(1, 11)
    plt.title(title)
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.plot(depths, training_errors, color="blue",
             marker='o', label='Training Errors')
    plt.plot(depths, testing_errors, color="red",
             marker='o', label='Testing Errors')
    plt.legend()
    plt.show()


def questionA(training_file, testing_file, title):
    # Load the training data
    M = np.genfromtxt(training_file, missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt(testing_file, missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depths 1-10 and plot a graph of the test error
    learning_curve(Xtrn, ytrn, Xtst, ytst, title)


def confusion_matrix_questionB(Xtrn, ytrn, Xtst, ytst, depth):
    decision_tree = id3(Xtrn, ytrn, max_depth=depth)
    visualize(decision_tree)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    totalCount = len(ytst)
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    for i in range(totalCount):
        if ytst[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif ytst[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif ytst[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif ytst[i] == 0 and y_pred[i] == 0:
            tn += 1
    l = [[tp, fn], [fp, tn]]
    print(l)
    print('            Confusion Matrix')
    print('           Positive   Negative')
    print('         |----------|----------|')
    print('         |          |          |')
    print('Positive |{0:6}    |{1:6}    |'.format(tp, fn))
    print('         |          |          |')
    print('         |----------|----------|')
    print('         |          |          |')
    print('Negative |{0:6}    |{1:6}    |'.format(fp, tn))
    print('         |          |          |')
    print('         |----------|----------|')


def questionB():
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    confusion_matrix_questionB(Xtrn, ytrn, Xtst, ytst, 1)
    confusion_matrix_questionB(Xtrn, ytrn, Xtst, ytst, 2)


def tree_and_confusion_matrix_questionC(Xtrn, ytrn, Xtst, ytst, filename):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Xtrn, ytrn)
    graph = graphviz.Source(tree.export_graphviz(
        clf, out_file=None, filled=True))
    graph.format = 'png'
    graph.render(filename, view=True)
    y_pred = clf.predict(Xtst)
    print("Sklearn's Confusion Matrix:")
    print(confusion_matrix(ytst, y_pred))


def questionC():
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    tree_and_confusion_matrix_questionC(
        Xtrn, ytrn, Xtst, ytst, "Scikit-Decision-Tree")

def unique(list1):
  
    # initialize a null list
    unique_list = []
  
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    return unique_list

def questionD():
    data = np.genfromtxt('./car.data', missing_values=0,
                         skip_header=0, delimiter=',', dtype=str)
    
    x = data[:, 0:len(data[0])-1]
    for i in range(len(data[0])-1):
        uniqueVals = unique(x[:, i])
        for j in range(len(x[:, i])):
            x[:, i][j] = uniqueVals.index(x[:, i][j]) + 1
    y = data[:, (len(data[0]) - 1)]
    # convert labels to binary labels: unacc = 0; acc, good, vgood = 1
    for i in range(len(y)):
        if y[i] != 'unacc':
            y[i] = 1
        else:
            y[i] = 0
    y = y.astype(np.int64)

    # Splits dataset into 70-30 training testing split
    ytst = y[int((len(data)+1)*.70):]
    Xtst = x[int((len(data)+1)*.70):]
    Xtrn = x[:int((len(data)+1)*.70)]
    ytrn = y[:int((len(data)+1)*.70)]
    confusion_matrix_questionB(Xtrn, ytrn, Xtst, ytst, 1)
    confusion_matrix_questionB(Xtrn, ytrn, Xtst, ytst, 2)
    tree_and_confusion_matrix_questionC(
        Xtrn, ytrn, Xtst, ytst, "Car-Data-Decision-Tree")


if __name__ == '__main__':
    questionA('./monks-1.train', './monks-1.test', "Monks 1 Learning Curve")
    questionA('./monks-2.train', './monks-2.test', "Monks 2 Learning Curve")
    questionA('./monks-3.train', './monks-3.test', "Monks 3 Learning Curve")
    questionB()
    questionC()
    questionD()
