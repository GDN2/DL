a=123
b="abc"+str(a)
print(b,type(b))

partcnt=0
part = "part" + str(partcnt)
print(part,type(part))

string="ABCDE"
cnt=0
for text in string :

    print(text,end="")
    cnt += 1
    if(cnt % 3) == 0 :
        print("")
        print(type(text))
print(len(string))