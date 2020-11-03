string = open("./petit prince.txt", 'rb').read().decode('UTF-8')
cnt=0
pagecnt=0
page=1
raw=11 #한 페이지 행 수 조금씩 다름
column=14 #한 페이지 열 수 조금씩 다름

path = "petit_prince/petit prince.split"+str(page)+".txt"
fw = open(path, 'w', encoding='UTF-8')
for text in string :
    print(text,end="")
    cnt += 1
    if(cnt % 13) == 0 :
        print("")
print(len(string))

for i in range(0,len(string)) :
    cnt = cnt + 1
    if string[i]=='\n':
        continue
    fw.write(string[i])

    if ((cnt % column) == 0): #한줄에 글자 수
        pagecnt +=1
        if((pagecnt % (raw)) == 0): #파트별 행 수 더 적게 적어야함
            fw.write("\npage"+str(page)+"\n")
            fw.close()
            page += 1
            path = "petit_prince/petit prince.split" + str(page) + ".txt"
            fw = open(path, 'w', encoding='UTF-8')
        else: fw.write("\n")

fw.close()
print("된거임?")