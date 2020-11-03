f = open("./file_test.txt", 'w')
f.write("hello, Python")
f.close()

with open("./file_test.txt", 'r') as f:
    print(f.read()+"!!!!!!")