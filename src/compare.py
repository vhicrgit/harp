import os
from utils import get_root_path
from os.path import dirname, abspath, getsize, abspath, join



# def get_sources(filePath):
#     allFile = []
#     for file in os.listdir(filePath):
#         allFile.append(file)
#     return allFile

def get_sources_size(filePath):
    allFile = dict()
    for file in os.listdir(filePath):
        allFile[file] = getsize(join(filePath, file))
    return allFile


def convert_line_endings(input_file, output_file):
    # 读取原始文件内容
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        content = infile.read()
    # 替换换行符
    new_content = content.replace('\r\n', '\n').replace('\r', '\n')
    # 写入新文件
    with open(output_file, 'w', newline='\n', encoding='utf-8') as outfile:
        outfile.write(new_content)

def formate_file(filePath):
    for file in os.listdir(filePath):
        if file.endswith('.c'):
            convert_line_endings(join(filePath, file), join(filePath, file))


def files_equal(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        return f1.read() == f2.read()


kaggleFilePath = os.path.join(get_root_path(), 'dse_database', 'kaggle', 'sources')
polyFilePath = os.path.join(get_root_path(), 'dse_database', 'poly', 'sources')
machsuiteFilePath = os.path.join(get_root_path(), 'dse_database', 'machsuite', 'sources')
formate_file(kaggleFilePath)
formate_file(polyFilePath)
kaggle = get_sources_size(kaggleFilePath)
poly = get_sources_size(polyFilePath)
machsuite = get_sources_size(machsuiteFilePath)

# print(f'len of kaggle files {len(kaggle)}')
# print(f'len of poly files {len(poly)}')
# print(f'len of machsuite files {len(machsuite)}')
# print(f'len of poly + machsuite files {len(poly.update(machsuite))}')
poly.update(machsuite)
print(sorted(poly.items()))
print(sorted(kaggle.items()))
print(sorted(poly)==sorted(kaggle))

print(files_equal(join(kaggleFilePath, '3mm_kernel.c'), join(polyFilePath, '3mm_kernel.c')))