import tensorflow as tf

# since onyl using cpu
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tensors
"""
# have a data type and a shape (representation/dimension of data)
# have ranks; rank = degree = num of dimensions


#string = tf.Variable("this is a string", tf.string)

# rank 1 tensor; 1 list/1 array/ 1 dimension
#rank1_tensor = tf.Variable(["Test", "Ok", "Tim"], tf.string);

# rank2 tensor;
#rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string);

# seeing the ranks

print(tf.rank(string))
print(tf.rank(rank1_tensor))
print(tf.rank(rank2_tensor))


# result is 

    tf.Tensor(0, shape=(), dtype=int32)
        0 = no rank
    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(2, shape=(), dtype=int32)


# seeing the shape; shape = amount of elements in each dimension
# print(rank2_tensor.shape)   # should print out 2 because 2 items in each array in rank2_tensor

    # output = (2, 2)
    
    # won't work if items in each dimensions aren't uniform (have to be same amt of items)
    
# print(rank1_tensor.shape)

    # output = 3 
    # there is only one dimension so prints out how many items in one dimension
"""

# changing shape
"""
tensor1 = tf.ones([1, 2, 3])

print(tensor1)

tensor2 = tf.reshape(tensor1, [2, 3, 1])
    # 2 lists, 3 lists inside each list, 1 element in each of the 3 lists
    
print(tensor2)

tensor3 = tf.reshape(tensor2, [3, -1])
    # 3 lists
    # -1 tells tensor to calculate size of dimension in that place
    # will reshape tensor to [3, 2]
        # 3 lists with 2 lists inside each of those 3
        
print (tensor3)
"""

# types of tensors
"""
# variable, constant, placeholder, sparsetensor
# except variable, all immutable; can't change during execution; all constant

# evaluating tensors = get a value
with tf.Session() as sess:
    tensor.eval()
        # tensor is the name of tensor
"""

# example
""""""

# tf.ones() creates all the values to be ones in whatever shape
    # tf.zeros() would be just a bunch of zeroes
t = tf.zeros([5, 5, 5, 5])
    # 5 lists with 5 lists in each of those with 5 lists in each of those, with 5 0's in each of those lists

print(t)

# flattening them out
t = tf.reshape(t, [625])
    # 625 0's in one list

t = tf.reshape(t, [125, -1])
    # 125 * 5 = -1
    # 5 sets of 5 how many times?

print(t)