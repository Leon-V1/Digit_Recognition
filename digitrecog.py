
"""
Created on Dec 29 2020

Digit Recognition Classification Project on MNIST database,

@author: Leon
"""

from sklearn import datasets, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm


#load data
digits = datasets.load_digits(as_frame=True)

#show first 5 images and target label
#for i in range(0,6):
#    plt.figure()
#    plt.imshow(digits.images[i], cmap=plt.cm.gray_r)
#    plt.title('Target: '+ str(digits.target[i]))
#    plt.show()

#reshape 8x8 -> 1x64      
data = digits.images.reshape((len(digits.images),-1))

#split data
X_train, X_test, y_train, y_test = train_test_split(data,digits.target, test_size = 0.2, shuffle=False)

#create classifiers
clf1 = KNeighborsClassifier(n_neighbors = 4)
clf2 = RandomForestClassifier(n_estimators=100)
clf3 = LinearSVC()
clf4 = svm.SVC(max_iter = 500000)

#fit classifiers on training data
clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)
clf4.fit(X_train,y_train)

#predict target value
predict1 = clf1.predict(X_test)
predict2 = clf2.predict(X_test)
predict3 = clf3.predict(X_test)
predict4 = clf4.predict(X_test)

A_KNN = metrics.accuracy_score(y_test, predict1)*100
A_RNGFor = metrics.accuracy_score(y_test, predict2)*100
A_LinearSVC = metrics.accuracy_score(y_test, predict3)*100
A_SVC = metrics.accuracy_score(y_test, predict4)*100

print('Analyze Classifier performance with accuracy_score: ')
print("KNN = " + str(round(A_KNN)), " RNG = " + str(round(A_RNGFor)), " L_SVC = " + str(round(A_LinearSVC)), " SVC = " + str(round(A_SVC)))

#plot some results
#for i in range(25,45):
#    plt.figure()
#    plt.imshow(digits.images[i], cmap=plt.cm.gray_r)
#    plt.title('Predict: '+ str(predict1[i]) + 'Actual: ' + str(digits.target[i]))
#    plt.show()

print(f"Classification report for classifier {clf1}:\n"
      f"{metrics.classification_report(y_test, predict1)}\n")

#We find that n_neighbors = 3,4 maximize accuracy at 97%.