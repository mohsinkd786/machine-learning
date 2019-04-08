# pip3 install numpy
# pip3 install matplotlib
# pip3 install pandas

import numpy as np

#twoDArray = np.array([1,2,4,3,8,10])
#print(twoDArray.shape)

# reshape
#mtx = twoDArray.reshape(3,2)
#print(mtx)

# shape 
#print(mtx.shape)

# zeroes
#print(np.zeros((2,2),dtype=float))

# ones
#print(np.ones((3,4),dtype=int))

# as array
#tpl = (5,4,12,9)
#print(np.asarray(tpl))

# arange
# index
# limit
# seed

rangeVals = np.arange(10,100,5)
#print(rangeVals.reshape(3,6))


# slicing
# start - index 
# seed - index
# skip 
#sliced = slice(2,10,2)
#print(rangeVals[sliced])

# 
slicedVals = rangeVals[1:10:1]
#print(slicedVals)

#slicedValsNoSkip = rangeVals[2:10]
#print(slicedValsNoSkip)

# ellipsis

arr = np.arange(0,9).reshape(3,3)
#print(arr)

# array elements in third column
#print(arr[...,2])

# array elements in second row
#print(arr[1,...])

# array of elements after first row i.e. 0 index
# shall be skipped
#print(arr[...,1:])

# matrix operations
#print(arr)
#arrTranspose = arr.T
# transpose
#print(arrTranspose)

randMatrx =np.array([9,3,5,4,8,1,7,6,2]).reshape(3,3)
#print(randMatrx)
#matx = randMatrx.copy(order = 'F')
#for n in np.nditer(matx):
#    print(n)

# flatten
# row based - C
# column based - F
#print(randMatrx.flatten(order = 'C'))

# flat index access
#print(randMatrx.flat[2])

# array concatenation
x = np.array([1,4,5,9]).reshape(2,2)
#print(x)
y = np.array([8,6,7,10]).reshape(2,2)
#print(y)

# row wise
z = np.concatenate((x,y))
#print(z)
# column wise
z = np.concatenate((x,y),axis=1)
#print(z)

# array stacking
# horizontal stacking
z = np.hstack((x,y))
#print(z)

# vertical stacking
z = np.vstack((x,y))
#print(z)

# product
z = x * y
#print(z)

# product by scalar
z = x * 3
#print(z)

# addition of matrices
z = x + y
#print(z)

# splitting
x = np.arange(1,25)
#print(x)

# split one array into 3 arrays
y = np.split(x,3)
#print(y)

# split using specific position
y = np.split(x,[5,10,15,20])
#print(y)

# statistical functions

# mean
x = np.array([[1,2,3],[3,4,5],[4,5,6]]) 

print(np.mean(x))

# along axis 0
# column wise
print(np.mean(x,axis = 0))

# along axis 1 
# row wise
print(np.mean(x,axis = 1))

# median
print(np.median(x))

# along axis 0
# column wise
print(np.median(x,axis = 0))

# along axis 1
# row wise
print(np.median(x,axis = 1))

# average
print(np.average(x))

# standard deviation
print(np.std(x))

# variance
print(np.var(x))

# dot / matrix multiplication
a = np.array([[1,2],[3,4]]) 
b = np.array([[11,12],[13,14]]) 

print(np.dot(a,b))

# algebra
x = np.array([[1,2], [3,4]]) 
print(np.linalg.det(x))

y = np.array([[11, 12], [13, 14]]) 

# inner
print(np.inner(a,b))


