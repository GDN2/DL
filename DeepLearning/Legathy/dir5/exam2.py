DATA = [[1,2,3,4],[10,20,30,40],[-1,-2,-3,-4]]

print(DATA)
X=[]
t=[]
y=[x[0:-1] for x in DATA]
for x in DATA:
    X.append(x[0:-1])
for x in DATA:
    t.append(x[-1])

print(X)
print(t)
print(y)

