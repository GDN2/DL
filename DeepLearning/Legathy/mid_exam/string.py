a = 'A73, CD'
print(a[1])

a = a + ', EFG'
print(a)

b = a.split(',')
print(b)
print(type(b), type(a), len(b), len(a))

for data in range(len(a)):
    print(a[data],end="")
print()
list_data = [10, 20, 30, 40, 50]
for data in list_data:
    print(data, "", end='')
print()

dict_data = {'key1':1, 'key2':2}
for data in dict_data:
    print(data,'',end="")
print()
for key, value in dict_data.items():
    print(key,"",value)

data = 5
while data >= 0:
    print("data ==", data)
    data -= 1
print(data)

data = 5
while data >= 0:
    print("data ==", data)
    data -= 1

    if data == 2:
        print("break here")
        break
    else:
        print("continue here")

