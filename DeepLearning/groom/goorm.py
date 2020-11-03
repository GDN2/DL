user_input = input()


paper_dict = {}
paper_list = []
paper_line_list = []
temp_list = []
count = 1
paper = []

string=''

try:
    print( user_input)
    paper_line_list = user_input.split(']')
    new_paper_line_list = user_input.split('.[ ')

    #for i in range(len(new_paper_line_list)):
     #   print(new_paper_line_list[i])

    for i in range(len(paper_line_list)-1):
        #print()
        #print(paper_line_list[i])
        temp_list = paper_line_list[i].split('[')
        paper_list.append(temp_list)
        paper_list[i] = paper_list[i][1].split(', ')


    for i in range(len(paper_list)):
        for j in range(len(paper_list[i])):
            paper_list[i][j] = paper_list[i][j].strip()
            if paper_list[i][j] in paper_dict:
                if paper_list[i][j] not in paper:
                    paper.append(paper_list[i][j])
            else :
                paper_dict[paper_list[i][j]] = count
                if paper_list[i][j] not in paper:
                    paper.append(paper_list[i][j])
                count += 1
    new_user_list = user_input.split(' ')
    for i in range(len(new_user_list)):
        if new_user_list[i].strip() in paper_dict:
            new_user_list[i] = str(paper_dict[new_user_list[i]])
        elif new_user_list[i].replace(',','').strip() in paper_dict:
            new_user_list[i] = str(paper_dict[new_user_list[i].replace(',','')])
            new_user_list[i] = str(new_user_list[i])+','
    for i in range(len(new_user_list)):
        temp_string = new_user_list[i]
        string += str(new_user_list[i])+" "

    print(string)
    for i in range(len(paper)):
        print("["+str(i+1)+"] "+paper[i])


except IndexError as err:
    print(str(err))



