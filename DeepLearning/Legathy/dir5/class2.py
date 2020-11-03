class Person:
    count = 0

    def __init__(self, name):
        self.name = name
        Person.count += 1
        print(self.name + " is initialized")

    def work(self, company):
        print(self.name + " is working on " + company)

    def sleep(self):
        print(self.name + " is sleeping")

    @classmethod
    def getCount(cls):
        return cls.count

obj1 = Person("GO")
obj2 = Person("PARK")

obj1.work("GOROKKE HOUSE")
obj2.work("MOVIE THEATER")

obj1.sleep()
obj2.sleep()

print(obj1.name)
print(obj2.name+"\n")
print("개행")

print(Person.getCount())
print(Person.count)
print(obj1.count)

class PrivateMemberTest:

     def __init__(self, name1, name2):

         self.name1 = name1
         self.__name2 = name2
         print("initialized with " + name1 + " ," +name2)

     def getNames(self):
         self.__PrintNames()
         return self.name1, self.__name2

     def __PrintNames(self):
         print(self.name1, self.__name2)

obj3 = PrivateMemberTest("GO", "PARK")

print(obj3.name1)
tuple1 = obj3.getNames()
print(obj3.getNames())
print(tuple1)
print(tuple1)