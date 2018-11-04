import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def px(val):
    print(val)
    print()

s = pd.Series([1,3,5,np.nan,6,8])
px(s)

dates = pd.date_range('20130101',periods=6)
px(dates)

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
px(df)

df2 = pd.DataFrame({    'A' : 1.,
                        'B' : pd.Timestamp('20130102'),
                        'C' : pd.Series(1,index=list(range(4)), dtype='float32'),
                        'D' : np.array([3] * 4, dtype='int32'),
                        'E' : pd.Categorical(["test", "train", "test", "train"]),
                        'F' : 'foo'})
px(df2)
px(df2.dtypes)
print(df.head(),df.tail(3),df.tail(4),df.index,sep='\n')
print(df.columns,df.values, df.describe(),sep='\n')
print(df.T)
print()
print(df.sort_index(axis=1, ascending=False))
print(df.sort_values(by='B'))
print(df['A'])
print(df[0:3])
print(df['20130102':'20130104'])
print(df.loc[dates[0]])
print(df.loc[:,['A','B']])
print(df.loc['20130102',['A','C']])
print(df.at[dates[0],'B'])
print(df.iloc[:,1:3])
print(df[df.A > 0])
print(df[df > 0])

df2 = df.copy()
df2['E'] = ['one','one','two','three','four','three']
print(df2)
print(df2[df2['E'].isin(['two','four'])])
df.iat[0,1] = 0
df.loc[:,'D']=np.array([5]*len(df))
print("here")
print("means",df.mean(),df.mean(1))


left  = pd.DataFrame({'key': ['foo','bar'],'lval':[1,2]})
right = pd.DataFrame({'key': ['foo','bar'],'rval':[4,5]})
print(left,right)

print(pd.merge(left,right,on='key'))

df = pd.DataFrame(np.random.randn(8,4), columns=['A','B','C','D'])
print(df)

s = df.iloc[3]
print(s)

df.append(s, ignore_index=True)
print(df)

df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
       'B' : ['A', 'B', 'C'] * 4,
       'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
       'D' : np.random.randn(12),
       'E' : np.random.randn(12)})
print(df)

newpt = pd.pivot_table(df, values='D', index=['A','B'], columns=['C'])
print(newpt)

rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts  = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(rng,ts, sep='\n')
print(ts.resample('5Min').sum())

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()

df = pd.DataFrame(np.random.randn(1000,4), index=ts.index,columns=['A','B','C','D'])
df = df.cumsum()
plt.figure()
df.plot()
plt.legend(loc='best')
plt.show()

df.to_csv('foo.csv')

loaddb = pd.read_csv('foo.csv')
print(loaddb)

df.to_hdf('foo.h5','df')
loaddbhdf = pd.read_hdf('foo.h5','df')
print(loaddbhdf)

df.to_excel('foo.xlsx', sheet_name='Sheet1')
loadxlsx = pd.read_excel('foo.xlsx','Sheet1', index_col=None,na_values=['NA'])
print(loadxlsx)
