import tensorflow as tf

""" Initialization of Tensors """
x = tf.constant(4 , shape = (1,1) , dtype = tf.int8)
print(x)
y = tf.constant([[1,2,3] , [4,5,6]])
print(y)

z = tf.ones((3 , 3)) # Creates a Tensor with specified dimension full of 1s
print(z)
a = tf.zeros((3 , 3) , dtype = tf.int8) # Take a guess
print(a)
b = tf.eye(3 , 3) # Creates an identity matrix/Tensor with specified dimensions
print(b)

c = tf.random.normal((3 , 3) , mean = 0 , stddev = 1)
print(c)
c = tf.random.uniform((3 , 3) , minval = 0 , maxval = 1)
print(c)
# delta is the same as step
c = tf.range(9 , delta = 2)
print(c)
c = tf.cast(c , dtype = tf.int8)
print(c)

""" MATH """
x = tf.constant([1 , 2 , 3])
y = tf.constant([9 , 8 , 7])
print(x + y)
