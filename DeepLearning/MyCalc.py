class MyCalc:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def sum(self):
        return self.x + self.y

    def subtract(self):
        return self.x - self.y

    def multiply(self):
        return self.x * self.y

    def divide(self):
        div = 0
        if self.y != 0:
            div = self.x / self.y
        return div


try:
    obj = MyCalc(1, 0)

    if (str(type(obj.x)) != "<class 'int'>") and (str(type(obj.x)) != "<class 'float'>"):
        raise Exception("First parameter is not number")
    elif (str(type(obj.y)) != "<class 'int'>") and (str(type(obj.y)) != "<class 'float'>"):
        raise Exception("Second parameter is not number")
    elif obj.y == 0:
        raise Exception("The denominator is zero!, It can't be divide")
    elif obj.x == float('inf') or obj.y == float('inf') or obj.x == float('-inf') or obj.y == float('-inf'):
        raise Exception("The number is over the range")
    else:
        print("sum =", obj.sum(), "subtract =", obj.subtract(), "multiply =", obj.multiply(), "divide =",
              obj.divide())

except Exception as err:
    print(str(err))

finally:
    pass
