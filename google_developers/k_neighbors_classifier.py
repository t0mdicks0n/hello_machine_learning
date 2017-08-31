# Below we implement a K nearest neighbour algo from scratch
# For specific problems KNN are sometimes able to do really well!
# But as you can see in the code it is quite slow (since it 
# for each test case have to iterate through all the training
# data to find the nearest neighbour).

from scipy.spatial import distance

def euc(a,b) :
	return distance.euclidean(a,b)

class ScrappyKNN() :
	def fit(self, x_train, y_train) :
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x_test) :
		predictions = []
		for row in x_test :
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self, row) :
		best_dist = euc(row, self.x_train[0])
		best_index = 0
		for i in range(1, len(self.x_train)) :
			dist = euc(row, self.x_train[i])
			if dist < best_dist :
				best_dist = dist
				best_index = i
		return self.y_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
# With the imported tool split the data on half for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)
# Import a classifier
from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()
# Define our classifier with our own instead
my_classifier = ScrappyKNN()
# Train the classifier with the data
my_classifier.fit(x_train, y_train)
# Make predictions
predictions = my_classifier.predict(x_test)
# Find out how well we did
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
# We did very well and was able to predict with 95% approx when tried!
