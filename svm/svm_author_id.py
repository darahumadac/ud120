#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

def print_prediction(prediction_array, indices):
	for i in indices:
		print "Prediction for #", i, ": ", prediction_array[i]

	print "\n"
	return

def get_label_count(prediction_array, label):
	import numpy as np
	return np.count_nonzero(prediction_array == label)

def process(clf):

	print "Training classifier..."
	t0 = time()
	clf.fit(features_train, labels_train)
	print "Training Time: ", round(time()-t0, 3), "s"
	print "Predicting test data..."
	p0 = time()
	prediction = clf.predict(features_test)
	print "Training Time: ", round(time()-p0, 3), "s"

	#Print prediction for 10th, 26th, and 50th
	print_prediction(prediction, [10, 26, 50])

	print "Computing accuracy score..."
	from sklearn.metrics import accuracy_score
	print "Accuracy score: ", accuracy_score(labels_test, prediction), "\n"

	#Count Cris emails
	print "Cris Labels: ", get_label_count(prediction, 1)


	return

#cut down the training dataset size to 1%
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


from sklearn.svm import SVC
print "Creating classifier..."

#classifier = SVC(kernel='linear')
#classifier = SVC(kernel='rbf')

#different C parameter values
classifier = SVC()

# print "C=10.0.."
# params_c10 = {'kernel': 'rbf', 'C':10.0}
# classifier.set_params(**params_c10)
# process(classifier)

# print "C=100.0.."
# params_c100 = {'kernel': 'rbf', 'C':100.0}
# classifier.set_params(**params_c100)
# process(classifier)

# print "C=1000.0.."
# params_c1000 = {'kernel': 'rbf', 'C':1000.0}
# classifier.set_params(**params_c1000)
# process(classifier)

print "C=10000.0.."
params_c10000 = {'kernel': 'rbf', 'C':10000.0}
classifier.set_params(**params_c10000)
process(classifier)

#########################################################


