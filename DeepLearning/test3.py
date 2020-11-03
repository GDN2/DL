import numpy as np

a = []
f = open('./data/test.csv', 'rb')
A = list(f.read().decode('UTF-8'))
print(f.read().decode('UTF-8'))
print(type(f))
f.close
print("A", A)
print("A", f.read().decode('UTF-8'))
b = np.array([['chef','noble',3],['normal', '5', 'dragon']], dtype='U')
print(b)
print(type(b))
print(b.dtype)
print(b)
print(b[1][2])
for index in range(2):
    for j in range(3):
        if str(b[index][j]) == 'chef':
            b[index][j] = 0
        elif str(b[index][j]) == 'noble':
            b[index][j] = 1
        elif str(b[index][j]) == 'normal':
            b[index][j] = 2
        elif str(b[index][j]) == 'dragon':
            b[index][j] = 3
        else:
            pass
print(b)
c = np.array(b, dtype=np.float32)

print(c)
print("-----------------")
d = np.loadtxt('./data/test3.csv', delimiter=',', dtype='U', encoding='UTF-8')
d[0][0] = 'chefc'
print(type(d[0][0]))
print(type(d[0][2]))
print("@@@@", d[0][2])
print(d)
for index in range(2):
    for j in range(3):
        if str(d[index][j]) == 'chefc':
            d[index][j] = 0
        elif str(d[index][j]) == 'noble':
            d[index][j] = 1
        elif str(d[index][j]) == 'normal':
            d[index][j] = 2
        elif str(d[index][j]) == 'dragon':
            d[index][j] = 3
        elif str(d[index][j]) == '':
            d[index][j] = 4
        else:
            pass
print(d)
c = np.array(d, dtype=np.float32)
print(d)
print("d[0]", d[0])
