
map = { 'One': 'John',
        'Two':'Bob',
        'Three':'Steve' }

# traverse via dictionary
for u in map :
    print(u)

# traverse via dictionary keys
lstKeys = list(map.keys())
print(lstKeys[0])

lst = '123'
lst = int(lst)
lst = float(lst)

# traverse via dictionary values
print(map.values())

print('Map Value by Key ####')
print(map['One'])

users = { 'One': ['John','Doe',12],
          'Two':['Bob','Samuels',22],
          'Three':['Adam','Rangers',31]
        }
for u in users.items() :
    #print(u)
    for val in u[1] :
        print(val)

emps = [
    {
        'name': 'John',
        'age' : 21,
        'designation': 'Programmer'
    },
    {
        'name': 'Doe',
        'age' : 25,
        'designation': 'Programmer'
    },
    {
        'name': 'Smith',
        'age' : 30,
        'designation': 'Manager'
    }
]
# traversing over list of employees with user objects 
for e in emps :
    # list of user names
    userNames = list(e['name'])
    print('Welcome Mr/Miss : ',e['name'],e['designation'])


# calculate P(A|B) 
def bayesTheorm(aProb,bProb):
    # P(B|A) = (P(A)x P(B)) / P(A)
    conditionalProbofBGivenA = (aProb * bProb) / aProb
    # Bayes Theorm
    # P(A|B) = (P(A) * P(B|A)) / P(B)
    return ((aProb) * conditionalProbofBGivenA) / bProb

print(bayesTheorm(10,20))