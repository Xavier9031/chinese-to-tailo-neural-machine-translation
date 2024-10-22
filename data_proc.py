import csv
import json
import argparse

# 創建一個parser來解析命令行參數
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/example_train.csv')   # 輸入文件
parser.add_argument('--output', type=str, default='data/example_train.json') # 輸出文件
args = parser.parse_args()  # 解析命令行參數

file_path = args.input  # 從命令行參數獲取輸入文件路徑

# 將每行轉換為指定格式的函數
def convert_row(row):
    return {
        'id': row['id'],
        'translation': {
            'zh': row['ZH'],  # 假設 'TL' 對應 'zh'
            'tl': row['TL']   # 假設 'ZH' 對應 'tl'
        }
    }

# 將轉換後的數據寫入新的 JSON 文件
output_file_path = args.output  # 從命令行參數獲取輸出文件路徑
converted_rows = []  # 存儲轉換後的行

# 再次讀取輸入文件並轉換每行
with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        converted_rows.append(convert_row(row))

# 將轉換後的數據寫入新的 JSON 文件
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(converted_rows, json_file, ensure_ascii=False, indent=4)
