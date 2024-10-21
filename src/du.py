import json

import pandas as pd

# 读取 CSV 文件
file_path = '/home/wqlou/kzw3933/harp/src/output.csv'
df = pd.read_csv(file_path)

# 检查 id 列是否有重复值
duplicate_ids = df[df.duplicated(subset=['id'], keep=False)]

if not duplicate_ids.empty:
    print("存在重复的 id:")
    print(duplicate_ids)
else:
    print("没有发现重复的 id")