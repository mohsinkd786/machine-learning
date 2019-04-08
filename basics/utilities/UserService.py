# user service class
class UserService:
    userId = 1120
    __uuid = 1119873636
    def __init__(self,firstName,lastName):
        self.firstName = firstName
        self.lastName = lastName
        
        print('User')
    
    def getUser(self,age):
        print('Hello ',self.firstName,age)
    
    def __getUuid(self):
        print('UUid ',__uuid)

uService = UserService('John','Doe')
uService.getUser(22)
print(uService.userId)
# private method
# print(uService.__getUuid())
# private variable
# print(uService.__uuid)
# print(uService.getUuid())

class Human :
    def __init__(self,ethinic):
        self.ethinic = ethinic
    
class EmployeeService(UserService,Human) :
    def __init__(self,firstName,lastName,ethinic):
        #self.firstName = firstName
        #self.lastName = lastName
        
        UserService.__init__(self,firstName,lastName)
        Human.__init__(self,ethinic)

        #print('Employee Service ',self.ethinic)
    
    def eDetails(self):
        print('Hello! ',self.firstName,self.lastName,self.ethinic)


eService = EmployeeService('John','Doe','Asian')
eService.eDetails()

sortedMatrix =[[1,2,3],
               [4,5,6],
               [7,8,9]]

inputMatrix =[[9,6,1],
              [5,3,7],
              [4,2,8]]


#inputMatrix.index([4,2,8])
