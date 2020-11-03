import datetime
start_time = datetime.datetime.now()
for i in range(10000000):
    end_time = datetime.datetime.now()
print(start_time)
print(end_time)
print(type(start_time))
print(end_time - start_time)