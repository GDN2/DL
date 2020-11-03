import glob
import os

print(os.getcwd())

all_files = glob.glob('/*'); #type을 리스트로 줌
print("ALL_FILES", all_files)
print("ALL_TYPES", type(all_files))

txt_files = glob.glob('/200203_DeppLearning/*.py')
print("All_txt_files", txt_files)