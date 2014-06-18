import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelBinarizer
# from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

## Reading the files..
X_train, y_train = load_svmlight_file("../data/wise2014-train.libsvm", dtype=np.float64, multilabel=True)
X_test, y_test = load_svmlight_file("../data/wise2014-test.libsvm", dtype=np.float64, multilabel=True)
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)

# ## Fitting the model and predicting..
# #clf = OneVsRestClassifier(SVC(kernel="poly"),n_jobs=-2)
# clf = OneVsRestClassifier(RandomForestClassifier(), n_jobs=-2)
# result = clf.fit(X_train, y_train)
# pred_y = clf.predict(X_test)

# ## Writing the output to a file..
# out_file = open("../submit/s5.RandomForest","w")
# out_file.write("ArticleId,Labels\n")
# id = 64858
# for i in xrange(pred_y.shape[0]):
# 	label = list(lb.classes_[np.where(pred_y[i,:]==1)[0]].astype("int"))
# 	label = " ".join(map(str,label))
# 	if label == "": 	## If the label is empty, populate the most frequent label
# 		label = "103"
# 	out_file.write(str(id+i)+","+label+"\n")
# out_file.close()
