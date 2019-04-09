import pandas as pd
import numpy as np

# empty series
#s = pd.Series()
#print (s)

# series data
s = pd.Series(np.arange(1,10))
#print(s)

# series with index
s = pd.Series([1,2,3,4],index = ['one','two','three','four'])
#print(s)

# key-value data
users = {   'John': { 'name':'John','phone': 123 },
            'Adam': { 'name': 'Adam', 'phone': 456 },
            'Steve': { 'name': 'Steve', 'phone': 779 }
        }

s = pd.Series(users)
#print(s)

# retrieve first 2 elements
#print(s[:2])

# retrieve via index
#print(s[1])

# retrieve last 2 elements
#print(s[-2:])

data = {'Name':['John', 'Bob', 'Dickins', 'Rocky'],'Age':[22,31,28,30]}
data1 = {'Name':['Adam', 'Simpsons', 'Rambo', 'Austen'],'Age':[11,55,27,25]}

dFrame = pd.DataFrame(data)
dFrame1 = pd.DataFrame(data1)

#print(dFrame)

# concatenate
dFrame = dFrame.append(dFrame1)
#print(dFrame1)

#print('Merged Data Frames ##### ')
#print(dFrame)

#print(dFrame[2:6])

# column selection
#print(dFrame['Name'])

# row location
#print(dFrame.loc['Name'])

# slicing
# print(dFrame[0:1])

# delete row
# via index
dFrame =dFrame.drop(0)
#print(dFrame)

# size 
#print(dFrame.size)

# series values
#print(dFrame.values)

# top rows
#print(dFrame.head(2))

# lower bound rows
#print(dFrame.tail(2))

d = {'Name':pd.Series(['John','Jack','Steve','Diesel','Sanders','Summer','Joly']),
   'Age':pd.Series([25,26,21,24,30,29,23]),
   'Ranking':pd.Series([1.9,3.24,3.98,2.56,3.20,4.6,3.8])}

# Create a DataFrame
df = pd.DataFrame(d)
#print(df)
#print("The transpose of the data series is:")
#print(df.T)

#### View all the methods

#print(dir(pd))
#for fn1 in dir(pd):
#    print(fn1)

# statistics

# sum
#print(df.sum())
# sum - axis : 1

#print(df.sum(1))

# mean
#print(df.mean())

# max
#print(df.max())

# average
#print(df.average())

# summary
#print(df.describe())

# for all the columns
#print(df.describe(include='all'))

# windowing
df = pd.DataFrame(np.random.randn(10, 2),
   index = pd.date_range('1/1/2000', periods=10),
   columns = ['A', 'B'])

#print(df)
# window sizing 
r = df.rolling(window=3,min_periods=1)
#print(r)
# mean
#print(df.rolling(window=4).mean())

# expanding
#print(df.expanding(min_periods=3).mean())

# aggregations
# sum
#print(r.aggregate(np.sum))

# sum - specific fields
#print(r['A'].aggregate(np.sum))

# multiple functions
# sum
# mean
#print(r[['A','B']].aggregate([np.sum,np.mean]))

# different aggregations on diff columns
#print(r.aggregate({'A' : np.sum,'B' : np.mean}))

# null values
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
#print(df)

#print(df['one'].isnull())
#print(df['one'].notnull())

# assumption : NA / NaN / Null values are treated as 0
#print(df['one'].sum())

# replace NA with a custom value
#print(df.fillna(0))

# drop NA
#print(df.dropna())

# drop NA 
# axis
#print(df.dropna(axis=1))

# replace scalars with specific values
df = pd.DataFrame({'one':[10,20,30,40,50,2300], 'two':[1020,0,30,40,50,60]})

#print(df.replace({1020:10,2300:60}))

# group by
# by key
# by multiple keys
# axis

iplTeams = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
   'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
   'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
   'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
   'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(iplTeams)

#print(df.groupby('Team'))

#print(df.groupby('Team').groups)

# group by multiple columns
#print(df.groupby(['Team','Year']).groups)

# traversing
# group by year
groupData = df.groupby('Year')

#for year,dataByYear in groupData:
#   print(year)
#   print(dataByYear)

# choose specific group value
#print(groupData.get_group(2017))


# aggregations
#print(groupData['Points'].agg(np.mean))

# size
# group by Teams
groupData = df.groupby('Team')
#print(groupData.agg(np.size))

# multiple aggregations
#print(groupData['Points'].agg([np.sum, np.mean, np.std]))

# condition
# only teams who have participated
# more than thrice
#print(df.groupby('Team').filter(lambda x: len(x) >= 3))

# merging
df1 = pd.DataFrame({
   'id':[1,2,3,4,5],
   'Name': ['John', 'James', 'Bobby', 'Andrews', 'Young'],
   'subject_id':['eng','math','physics','chemistry','biology'],
   'Marks_scored':[98,90,87,69,78]},
   index=[1,2,3,4,5])

df2 = pd.DataFrame({
	'id':[1,2,3,4,5],
   'Name': ['Brandon', 'Fowler', 'Rolex', 'Tommy', 'Johnson'],
   'subject_id':['math','physics','chemistry','hindi','biology'],
   'Marks_scored':[89,80,79,97,88]},
   index=[1,2,3,4,5])

#print(pd.merge(df1,df2,on='id'))

# multiple keys
#print(pd.merge(df1,df2,on=['id','subject_id']))

# how 
# joins
# left
# right
# outer
# inner
#print(pd.merge(df1, df2, on='subject_id', how='left'))

#print(pd.merge(df1, df2, on='subject_id', how='right'))

# concatenate

#print(pd.concat([df1,df2],keys=['x','y']))

# time delta
# time based deviations
s = pd.Series(pd.date_range('2012-1-1', periods=3, freq='D'))
td = pd.Series([ pd.Timedelta(days=i) for i in range(3) ])
df = pd.DataFrame(dict(A = s, B = td))

#print(df)
# addition
df['C']=df['A']+df['B']
#print(df)


# plots
df = pd.DataFrame(np.random.randn(10,4),index=pd.date_range('1/1/2000',
   periods=10), columns=list('ABCD'))

#df.plot()

# bar plot
df = pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d'])
#df.plot.bar()

# horizontal bars
#df.plot.barh(stacked=True)

# histogram
df = pd.DataFrame({'a':np.random.randn(1000)+1,'b':np.random.randn(1000),'c':
np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])

#df.plot.hist(bins=20)

# column based histograms
df=pd.DataFrame({'a':np.random.randn(1000)+1,'b':np.random.randn(1000),'c':
np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])

#df.diff.hist(bins=20)

# box plot
df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
#df.plot.box()

# scatter plot
df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
#df.plot.scatter(x='a', y='b')

# pie chart
df = pd.DataFrame(3 * np.random.rand(4), index=['a', 'b', 'c', 'd'], columns=['x'])
#df.plot.pie(subplots=True)

# read csv
df=pd.read_csv("users.csv")
#print(df)

# skip header
#df=pd.read_csv("users.csv",names=['a','b','c','d','e'],header=0)
#print(df)

#df=pd.read_csv("users.csv", skiprows=2)
print(df)

