score = {'kim':90, 'lee':85, 'go':100}

print(score)

print(score.keys())
print(score.values())
print(list(score.keys()))

a = [1,2,3,4,5,6,7,888]

for i in range(len(a)):
    print(a[i],"",end='')
for i in a:
    print(i,"",end='')