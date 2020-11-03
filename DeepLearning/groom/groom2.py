input_string = input()
sum = 0
while(1):
    try:
        input_int = int(input_string)
        if input_int < 2 or input_int > 20:
            raise Exception
        for i in range(int(input_string)):
            sum += i
        print(sum)
        break
    except Exception as err:
        input_string = ''
        print("2이상 20이하의 양의 정수를 입력해주세요.")
