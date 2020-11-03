import glob
import os

print(os.getcwd())

all_files = glob.glob('*.*'); #type을 리스트로 줌 파이참에서는 이상하게 나옴 쥬피터에서는 제대로 나옴
print("ALL_FILES", all_files)
print("ALL_TYPES", type(all_files))

txt_files = glob.glob('/*.py')
print("All_txt_files", txt_files)