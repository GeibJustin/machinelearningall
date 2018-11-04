import numpy as np

def printex(what_to_print):
    print(what_to_print)
    print()

a = np.array([1,2,3])
print("1x3:")
print(a)
print()

a = np.array([[1, 2], [3, 4]])
print("2x2: ")
print(a)
print()

a = np.array([1,2,3,4,5],ndmin = 2)
print("min dimensions 2: ")
print(a)
print()

a = np.array([1,2,3], dtype = complex)
print("complex num: ")
print(a)
print()

dt = np.dtype(np.int32)
print("unsigned int 32bit")
print(dt)
print()

print("int8, int16 etc, can be replaced by an equivalent string i1, i2 etc...")
dt = np.dtype('i1')
print(dt)
dt = np.dtype('i2')
print(dt)
dt = np.dtype('i4')
print(dt)
print()

print("can use endian notation")
print(np.dtype('>i4'))
print()

print("creating structured data type")
dt = np.dtype([('age', np.int8)])
printex(dt)

print("use data types in an ndarray")
dt = np.dtype([('age', np.int8)])
a  = np.array([(10,),(20,),(30,)], dtype = dt)
printex(a)

print("use last example to print out columns")
printex(a['age'])

print("student exmaple, string field name, integer field age, and float field marks")
student = np.dtype([('name','S20'),('age','i1'),('marks','f4')])
printex(student)

print("enter values using last dtype")
a = np.array([('firstlier the second',21,50.0),('名前',30,75.8)])
printex(a)

print("finding the shape/dim of an array")
printex(a.shape)

print("changing the shape")
a.shape = (3,2)
print(a.shape)
printex(a)

print("changing the shape using reshape (only temporary here need to reasign not an inplace function)")
print(a.reshape(2,3))
printex(a.shape)

print("evenly spacing numbers")
a = np.arange(24)
printex(a)

print("reshaping, divisible dimensions")
print("reshaping non-divisible dimensions NOT POSSIBLE test, failed...")
print(a.ndim)
b = a.reshape(2,4,3)
print(b.ndim)
printex(b)

print("using item size, returns 1, as describes length of each elem in bytes")
x = np.array([1,2,3,4,5], dtype = np.int8)
printex(x.itemsize)

print("current flag values: descriptions of array and important aspects of it")
printex(x.flags)

print("empty array...")
x = np.empty([3,2],dtype=int)
printex(x)

print("array of zeros, default is float")
x = np.zeros(5)
printex(x)

print("array of zeros as int, must specify!!!")
x = np.zeros((5), dtype = np.int)
printex(x)

print("array zeros, with custom types")
x = np.zeros((2,2), dtype = [('x', 'i4'), ('y','i4')])
printex(x)

print("new array filled with 1's (default float)")
x = np.ones(5)
printex(x)

print("array of ones int dtype")
x = np.ones([2,2], dtype=int)
printex(x)

print("convert list to ndarray, .asarray")
x = [1,2,3]
a = np.asarray(x)
printex(a)

print("dtype is set")
a = np.asarray(x, dtype = float)
printex(a)

print("ndarray from tuple, same func")
x = (1,2,3)
a = np.asarray(x)
printex(a)

print("list of tuples too!")
x = [(1,2,3),(4,5)]
a = np.asarray(x)
printex(x)
