# Import test data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Import tensorflow
import tensorflow as tf
# A symbolic variable
x = tf.placeholder(tf.float32, [None, 784])
# We create a Weight and bias input for our model
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# Implement the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
# We then implement our cross-entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Train the model given the loss minimizing cross entropy that we defined above
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# Launch the model
sess = tf.InteractiveSession()
# We create an operation to initialize the variables we created
tf.global_variables_initializer().run()
# Now we train it by running the training steps 1000 times
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
"Each step of the loop we get a batch of one hundred random data points from our training set"
"We then run train_step feeding in the batches data to replace the placeholders."

"We now want to evaluate our model."
# We use tf.equal to check if our prediction matches the truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# This gives us a list of booleans, we cast to floating point numbers and take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# We then finally ask for our accuracy on our test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

" We get around 92 percent accuracy! Which is not that good.."
