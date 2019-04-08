class HelloService:
    message = 'Hello How r u'

    def __init__(self):
        print('Hello Service')
    
    def getMessage(self):
        return self.message
    
    @staticmethod
    def getStaticMessage():
        return HelloService.message
    
service = HelloService()
print(HelloService.message)
# static method call
print(HelloService.getStaticMessage())
