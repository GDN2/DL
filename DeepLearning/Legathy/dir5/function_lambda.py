def sum(x,y):

    s = x+ y
    return s
result = sum(10,20)
print(result)

def multi_ret_func(x):
    return x+1,x+2,x+3
x=100
y = multi_ret_func(x)
print(y)
print(type(y))

def print_name(name, count=2):
    for i in range(count):
        print("name == ", name)
print_name('DAVE')

def mutable_immutable_func(int_x, input_list):
    int_x +=1
    input_list.append(100)
x = 1
test_list = [1,2,3]

mutable_immutable_func(x, test_list)
print("x == ", x, ", test_list ==", test_list)

f = lambda x : x+100


for i in range(3):
    print(f(i))

def ff(x):
    return x+100

for i in range(3):
    print(ff(i))