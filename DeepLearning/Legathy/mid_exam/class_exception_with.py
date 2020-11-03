class Person:
    count = 0

    def __init__(self, name):
        self.name = name
        Person.count += 1
        print(self.name + " is initailized")

    def work(self, company):
        print(self.name + " is working in " + company)

    def sleep(self):
        print(self.name + " is sleeping")

    @classmethod
    def getCount(cls):
        return cls.count
obj1 = Person("Park")
obj2 = Person("Kim")

obj1.work("ABCDEF")
obj2.sleep()

print("current person object is " + obj1.name + ", " + obj2.name)
print("Person count ==", Person.getCount())
print(Person.count)
print(obj1.count)

class PrivateMemberTest:
    def __init__(self, name1, name2):
        self.name1 = name1
        self.__name2 = name2
        print("initalized with " + name1 + name2)
    def getNames(self):
        self.__printNames()
        return self.name1, self.__name2
    def __printNames(self):
        print(self.name1, self.__name2)

obj = PrivateMemberTest("GO", "PARK")

print(obj.name1)
print(obj.getNames())

def print_name(name):
    print("[def] ", name)
class SameTest:
    def __init__(self):
        pass
    def print_name(self, name):
        print("[SameTest] ", name)
    def call_test(self):
        print_name("KiM")
        self.print_name("KIM")
obj = SameTest()
print_name("LEE")
obj.print_name("LEE")
obj.call_test()

def calc(list_data):
    sum = 0
    try:
        sum = list_data[0] + list_data[1] + list_data[2]

        if sum < 0:
            raise Exception("Sum is minus")

    except IndexError as err:
        print(str(err))
    except Exception as err:
        print(str(err))
    finally:
        print(sum)
calc([1,2])
calc([1,2,-300])

f = open("dir5/file_test.txt", 'w')
f.write("HELLLLLLLL")
f.close()
f = open("./file_test.txt", 'r')
print(f.read())
f.close()

with open("./file_test2.txt", 'w') as f:
    f.write("GEEEGGE")
with open("./file_test2.txt", 'r') as f:
    print(f.read())
