import dir5.class2

class Person:

    def __init__(self, name):
        self.name = name
        print(self.name + " is initialized")

    def work(self, company):
        print(self.name + " is working in " + company)

    def sleep(self):
        print(self.name + " is sleeping")

obj = Person("GO")

obj.work("GOROKKE HOUSE")
obj.sleep()
print(obj.name)

print(dir5.class2.obj3.getNames())