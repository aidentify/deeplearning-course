__author__ = 'socurites'

import tensorflow as tf

# create constants
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

# create session
sess = tf.Session()
print(sess.run([node1, node2]))

# combine nodes with operation add
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))

# create placeholders and add operation
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

# evaluate
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

# add operation
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b:4.5}))

# create variable for linear model: W * x + b
x = tf.placeholder(tf.float32)
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
linear_model = W * x + b

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(W))
print(sess.run(b))

# evaluation with x = [1,2,3,4]
print(sess.run(linear_model, {x:[1,2,3,4]}))

# create placeholder for real output
y = tf.placeholder(tf.float32)

# define loss: squared error
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# evaluate loss
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# assign values for W and b, then see the loss is 0
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# define optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# reset values to incorrect defaults
sess.run(init)

# run training repeatedly
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:[1,2,3,4], y:[0,-1,-2,-3]})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
