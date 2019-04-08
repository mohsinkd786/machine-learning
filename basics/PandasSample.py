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
print(dFrame)



