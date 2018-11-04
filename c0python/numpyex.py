import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt

def printex(what_to_print):
    print(what_to_print)
    print()

def px(w_t_print):
    printex(w_t_print)

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

print("From buffer function .frombuffer, doesn't work with py3")
print()

print("test iterator object from list")
list = range(5)
it   = iter(list)
x = np.fromiter(it, dtype = float)
printex(x)

print("Arrays from numerical ranges - .arange(start, stop, step, dtype)")
x = np.arange(5)
t = np.arange(5, dtype = float)
y = np.arange(10,20,2)
print(x,t,y,sep="\n")
print()

print("linspace: (start, stop, num, endpoint, retstep, dtype)")
print("num - number of evenly spaced samples default is 50")
print("endpoint=false, don't include end value of iter")
print("retstep returns samples and step between consec nums")
x = np.linspace(10,20,5)
px(x)

print("endpoint ex, 4 values")
y = np.linspace(10,20,5,endpoint=False)
px(y)

print("retstep example, returns step val ie slope")
z = np.linspace(1,2,5,retstep=True)
px(z)

print("logspace - sweet :), default base 10 :(")
a = np.logspace(1.0,2.0,num=10)
px(a)

print("base 2")
a = np.logspace(1,10,num=10, base=2)
printex(a)

print("slicing!, can do as an input")
a = np.arange(10)
s = slice(2,7,2)
px(a[s])

print("single slice item, ie index")
print(a[5])
px(a[2:])

# array to begin with
a = np.array([[1,2,3],[3,4,5],[4,5,6]])

print('Our array is:')
print(a)
print('\n')

# this returns array of items in the second column
print('The items in the second column are:')
print(a[...,1])
print('\n')

# Now we will slice all items from the second row
print('The items in the second row are:')
print(a[1,...])
print('\n')

# Now we will slice all items from column 1 onwards
print('The items column 1 onwards are:')
print(a[...,1:])

print("integer indexing")
x = np.array([[1,2],[3,4],[5,6]])
y = x[[0,1,2],[0,1,0]]
px(y)

print("Our  array")
x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
print(x)
rows = np.array([[0,0],[3,3]])
cols = np.array([[0,2],[0,2]])
y = x[rows,cols]
px(y)

print("Slicing")
z = x[1:4,1:3]
printex(z)

print("Slicing adv index for column")
y = x[1:4,[1,2]]
printex(y)

print("NaN use ~")
a = np.array([np.nan,1,2,np.nan,3,4,5])
print(a[~np.isnan(a)])
printex(a[np.iscomplex(a)])

print("Broadcasting - treating arrays of different sizes")
a = np.array([1,2,3,4])
b = np.array([10,20,30,40])
c = a * b
printex(c)

print("nparray")
a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]])
b = np.array([1.0,2.0,3.0])
print(a)
print(b)
print('a+b:')
printex(a+b)

print('array iteration using .nditer')
a = np.arange(0,60,5)
a = a.reshape(3,4)
print(a)
for x in np.nditer(a):
    print(x)

print('f-style order')
print("Transpose of original")
b = a.T
print(b)

print("c-style sort order")
c = b.copy(order='C')
print(c)
for x in np.nditer(c):
    print(x)
print()

print('f-style sort order')
c = b.copy(order='F')
print(c)
for x in np.nditer(c):
    print(x)
print()

a = np.array([0,30,45,60,90])
print(np.sin(a*np.pi/180))
print(np.cos(a*np.pi/180))
print(np.tan(a*np.pi/180))
print()

print("Matrix operations")
a = np.arange(9, dtype = np.float).reshape(3,3)
b = np.array([10,10,10])
print(a)
print(b)
print('a+b')
print(np.add(a,b))
print('a-b')
print(np.subtract(a,b))
print('ab')
print(np.multiply(a,b))
print('a/b')
print(np.divide(a,b))
print()

a = np.array([0.25,1.33,1,0,100])
print('Array:')
print(a)

print('Reciprocal function')
print(np.reciprocal(a))

b = np.array([100], dtype = int)
print('the second array is:')
print(b)
print(np.reciprocal(b))
print()

print("power function")
a = np.array([10,100,1000])
print(a)
print(np.power(a,2))
b=np.array([1,2,3])
print(b)
print(np.power(a,b))

print("numpy mod")
a = np.array([10,20,30])
b = np.array([3,5,7])
print(a)
print(b)
print(np.mod(a,b))
print(np.remainder(a,b))

a = np.array([[30,40,70,0],[80,20,10,0],[50,90,60,40]])

print("Percentile function")
print(np.percentile(a,50))
print(np.percentile(a,50, axis = 1))
print(np.percentile(a,50, axis = 0))
print()

print("Median")
printex(np.median(a))

print("Mean")
printex(np.mean(a))

print('var andstd dev')
print(np.var(a))
printex(np.var(a)**.5)

print('sort functions; .sort(a,axis, kind, order), kind default = quicksort')
print(np.sort(a, kind='mergesort'))

print('argsort(), returns indicese')
x = np.array([3,1,2])
print(np.argsort(x))
print()

print('lexsort')
nm = ('raju','anil','ravi','amar')
dv = ('f.y.','s.y.','s.y.','r.y.')
ind = np.lexsort((dv,nm))
print(ind)
print([nm[i] +', ' + dv[i] for i in ind])

print('Shallow copy; .view()')
a = np.arange(6).reshape(3,2)

print(a)
b = a.view()
print(b)

print('ids for a and b resp: ', id(a), id(b))
print()

print('Deep copy; .copy()')
c = a.copy()
printex(c)

print('matlib(), empty fills with rand')
printex(np.matlib.empty((2,2)))

print(np.matlib.eye(n = 3, M = 4, k = 0, dtype = float))
print()

print('random matrix entries')
print(np.matlib.rand(10,10)*10)

print('lin alg')
x = np.arange(1,11)
y = 2 * x + 5
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x,y)
plt.show()

x = np.arange(1,11)
y = 2 * x + 5
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x,y,"ob")
plt.show()

x = np.arange(0, 3*np.pi, 0.1)
y = np.sin(x)
plt.title("sine wave form")
plt.plot(x,y,"g")
plt.show()

x = np.arange(0,3*np.pi,0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(2,1,1)

plt.subplot(2,1,1)

plt.plot(x,y_sin)
plt.title('sine')

plt.subplot(2,1,2)
plt.plot(x,y_cos)
plt.title('Cosine')

plt.show()


x = [5,8,10]
y = [12,16,6]

x2 = [6,9,11]
y2 = [6,15,7]
plt.bar(x,y,align='center')
plt.bar(x2,y2,color='g',align='center')
plt.title('Bar graph')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()

a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
np.histogram(a,bins=[0,20,40,60,80,100])
hist,bins = np.histogram(a,bins = [0,20,40,60,80,100])
print(hist)
print(bins)
plt.hist(a,bins=[0,20,40,60,80,100])

plt.title('histogram')
plt.show()

np.save('outfile',a)
