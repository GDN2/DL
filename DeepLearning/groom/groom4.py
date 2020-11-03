
name_list = []

while(True):
    try:
        user_input = input()
        name_list = []
        for i in range(int(user_input)):
            name = input()
            name_split = name.split(' ')

            if len(name_split) != 2:
                raise Exception

            name_list.append(name_split)
        #print(name_list[0])
        #print(name_list[1])
        for i in range(len(name_list)):
            name_list[i][0] = name_list[i][0].lower().capitalize()
            name_list[i][1] = name_list[i][1].lower().capitalize()

            #print(name_list[1][0], name_list[i][1])

        for i in range(len(name_list)):
            print("Case #" + str(i+1))
            print(name_list[i][0] + " " + name_list[i][1])
        break
    except Exception as err:
        print(str(err))


'''
user_input = input()
name_list = []
for i in range(int(user_input)):
    name = input()
    name_split = name.split(' ')

    if len(name_split) != 2:
        raise Exception

    name_list.append(name_split)
print(name_list[0])
print(name_list[1])
for i in range(len(name_list)):
    name_list[i][0] = name_list[i][0].lower().capitalize()
    name_list[i][1] = name_list[i][1].lower().capitalize()

    print(name_list[1][0], name_list[i][1])

for i in range(len(name_list)):
    print("Case #"+str(i))
    print(name_list[i][0] + " " + name_list[i][1])
'''
