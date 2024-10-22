import torch
import pandas as pd
import tqdm
import csv
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import argparse

# 創建一個parser來解析輸入參數
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='my_awesome_HW3_bart-large_fine_tune/checkpoint-90444') # 訓練好的模型
parser.add_argument('--input', type=str, default='data/test-ZH-nospace.csv') # 輸入文件
parser.add_argument('--output', type=str, default='data/test_ans_bart_large_finetune.csv') # 輸出文件
args = parser.parse_args()

# 從指定的檢查點加載 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)

# 讀取輸入的 CSV 文件
test = pd.read_csv(args.input)

# 將測試數據中的文本轉換為模型可接受的格式
ZH_ids = []
for row in tqdm.tqdm(test.itertuples(index=False)):
    ZH_ids.append(tokenizer(row[1], return_tensors="pt").input_ids)

# 開始進行模型預測並將結果寫入到 CSV 文件中
with open(args.output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'txt'])
    for i in tqdm.tqdm(range(len(ZH_ids))):
        # 生成預測文本
        outputs = model.generate(ZH_ids[i], max_new_tokens=100, do_sample=True, top_k=30, top_p=0.95)
        id = test.iloc[i]['id']
        txt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        writer.writerow([id, txt])

