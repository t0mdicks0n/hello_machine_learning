from sklearn import tree 
# Input to the classifier:
# features = [[140, 'smooth'], [130, 'smooth'], [150, 'bumpy'], [170, 'bumpy']]
"As ints instead"
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# The output that we want:
# labels = ['apple', 'apple', 'orange', 'orange']
"As ints instead"
labels = [0, 0, 1, 1]

# Create the classifier
clf = tree.DecisionTreeClassifier()
# Learning algorithm
clf = clf.fit(features, labels)
# Test run the classifier
print clf.predict([[160, 0]])

# By comparing the above input to the training data
# we can intuitively guess that the program should
# predict the input to be a orange since its on the
# heavier side and bumpy
