PAPER = "AAA AAA AAA.[ some_paper_a, some_paper_b ] BBB BBB BBB.[ some_book_a, some_paper_a ] CCC CCC CCC.[ some_book_b ]"

user_input = input()
#user_input.replace(',', '')
paper_line =  user_input.split(']')
paper_list = []
paper_dict = {}
paper_num = 1
print(paper_line)
index_list = []
num_list = []

for i in range(len(paper_line)-1):
    temp_line = []
    temp_line = paper_line[i].split('[')
    index_list.append("["+temp_line[1]+"]")
    temp_line = temp_line[1].split(',')
    for i in range(len(temp_line)):
        if temp_line[i].replace(',', '').strip() not in paper_list:
            paper_list.append(temp_line[i].replace(',', '').strip())
            paper_dict[temp_line[i].replace(',', '').strip()] = str(paper_num)
            paper_num += 1

print(paper_list)
print(paper_dict.items())
print(index_list)

for i in range(len(paper_line)-1):
    temp_line = []
    temp_line = paper_line[i].split('[')
    index_list.append("["+temp_line[1]+"]")
    temp_line_2 = temp_line[1].split(',')
    for j in range(len(temp_line_2)):
        if temp_line_2[j].replace(',', '').strip() in paper_list:
            temp_line_2[j] = paper_dict[temp_line_2[j].replace(',', '').strip()]
    temp_line_2.sort()
    num_list.append(temp_line_2)

for i in range(len(num_list)):
    temp_line = []
    temp_string = ''
    temp_line = paper_line[i].split('[')
    temp_string = temp_line[0] + "[ "

    for j in range(len(num_list[i])):
        temp_string = temp_string + num_list[i][j]
        if j != len(num_list[i])-1:
            temp_string = temp_string+', '

    temp_string = temp_string + " ]"
    paper_line[i] = temp_string


print_string = ''
for i in range(len(paper_line)-1):
    print_string = print_string + paper_line[i]
print(print_string)

for i in range(len(paper_list)):
    print("[" + str(i + 1) + "] " + paper_list[i])