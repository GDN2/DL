list_data = [x**2 for x in range(5)]

print(list_data)

raw_data = [[1,10],[2,15],[3,30],[4,55]]

all_data = [x for x in raw_data]
x_data = [x[0] for x in raw_data]
y_data = [x[1] for x in raw_data]

print(all_data)
print(x_data)
print(y_data)

even_number = []

for data in range(10):
    if data % 2 ==0:
        even_number.append(data)
print(even_number)

data = 5
while data >= 0:
    print("data ==", data)
    data -= 1

data = 5
while data >= 0:
    print("data ==",data)
    data -= 1

    if data == 2:
        print("break here")
        break
    else:
        print("continue")
