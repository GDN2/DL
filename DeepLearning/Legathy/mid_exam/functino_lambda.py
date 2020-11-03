def multi_ret_func(x):
    return x+1, x+2, x+3

x = 100
y1 = multi_ret_func(x)
print(y1, type(y1))

def print_name(name, count=2):
    for i in range(count):
        print("name ==", name)
print_name("DAVE", 10)

def mmm(int_x, input_list):
    int_x += 1
    input_list.append(100)

x = 1
test_list = [1,2,3]

mmm(x, test_list)

print("x ==", x, ", test_list ==", test_list)

f = lambda x : x**5

for i in range(3):
    print(f(i))