import numpy as np
import pandas as pd

string = '../data/Thoracic.csv'
print(len(string))
tmp_index = 0
for index in range(len(string)):
    if string[len(string)-1-index] == '/':
        tmp_index = len(string)-1-index
        break
data_path = string[:tmp_index]
data_name = string[tmp_index+1:]
print(data_path)
print(data_name)