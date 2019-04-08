print('Hello')
i = 10
print('Value of I is ',i)

if i > 9 :
    print('I is greater than 9 ',i)
elif i == 3:
    print('I is 3')
else:
    print('I is smaller')

print('While Loop')

_index = 0
while _index < 10 :
    print(_index)
    _index = _index + 1

# for loop
print('For Loop')

# range(startIndex,endIndex,seed)

for k in range(4,-1,-2):
    print(k)

def sayHello() :
    print('Hello')

# call sayHello() method
sayHello()

# add
def add(a,b):
    return a + b

# call add method
print(add(10,2))

def process(i,j=10):
    return i + j

result = process(j=2,i=13)
print(result)
_calculate = lambda i,j : i * j
print(_calculate(5,2))
# custom calculator using lamda functions
def myCustomCalculator(_firstNum,_action) :
    if _action == 'ADD' :
        return lambda _nextNum : _firstNum + _nextNum
    elif _action == 'DIFF' :
        return lambda _nextNum : _firstNum - _nextNum
    elif _action == 'MUL' :
        return lambda _nextNum : _firstNum * _nextNum
    else :
        return 0

operate = myCustomCalculator(15,'MUL')
print('myCustomCalculator() Lambda ')
print(operate(10))

# list
users = ['John','Bob','Roger','Steve']
# add a new element
users.append('Serena')
# insert at a specific index
users.insert(0,'Agassi')

# iterate on list values
for u in users :
    if(u == 'John') :
        print('First ', users.index(u))
    print(u)

# list / tuple / dictionary

print(users[1])
print(users[1:3])


# modify the value
users[1] = 'Adams'
# remove a value from the list
del users[0]
print(users[0:-1])
# tuple
usersTuple = ('Chang','Chow','Missi')
# iterate on tuple
for u in usersTuple:
    print(u)
# will throw an error here , tuple is read only
# usersTuple[0] = 'Munchow'
# print(usersTuple[0])


from messages import getMessage,test

print(getMessage('Hey!'))
test()

from utils.mathutils import add as mAdd

print(mAdd(100,253))

