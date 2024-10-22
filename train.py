import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse

# 創建parser
parser = argparse.ArgumentParser()
# 添加需要的參數
parser.add_argument('--pretrained_model', type=str, default='facebook/bart-large') # 預訓練模型
parser.add_argument('--input', type=str, default='data/example_train.json') # 輸入文件
parser.add_argument('--max_length', type=int, default=128) # 最大長度
parser.add_argument('--output_dir', type=str, default='my_awesome_HW3_bart-large_fine_tune') # 輸出文件
parser.add_argument('--save_strategy', type=str, default='epoch') # 儲存策略
parser.add_argument('--evaluation_strategy', type=str, default='epoch') # 評估策略
parser.add_argument('--learning_rate', type=float, default=2e-5) # 學習率
parser.add_argument('--per_device_train_batch_size', type=int, default=2) # 訓練批次大小
parser.add_argument('--per_device_eval_batch_size', type=int, default=2) # 測試批次大小
parser.add_argument('--weight_decay', type=float, default=0.01) # 權重衰減
parser.add_argument('--save_total_limit', type=int, default=10) # 儲存總數
parser.add_argument('--num_train_epochs', type=int, default=10) # 訓練輪數
parser.add_argument('--predict_with_generate', type=bool, default=True) # 預測
parser.add_argument('--logging_dir', type=str, default='logs') # 日誌文件
parser.add_argument('--fp16', type=bool, default=True) # fp16
parser.add_argument('--push_to_hub', type=bool, default=False) # 推送到hub
args = parser.parse_args()

# 載入數據集
dataset = load_dataset('json', data_files=args.input)
# 切分數據集為訓練集和測試集
dataset = dataset["train"].train_test_split(test_size=0.001, shuffle=True, seed=42)

checkpoint = args.pretrained_model
# 初始化 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 設定原始語言和目標語言
source_lang = "zh"
target_lang = "tl"
prefix = "translate Chinese to Tailo: "

# 數據預處理函數
def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

# 應用數據預處理
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 初始化數據整理器
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# 載入評估指標
metric = evaluate.load("sacrebleu")

# 文本後處理函數
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

# 計算指標函數
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# 初始化模型
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
model.config.max_length = args.max_length

# 設置訓練參數
training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    save_strategy=args.save_strategy,
    evaluation_strategy=args.evaluation_strategy,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    weight_decay=args.weight_decay,
    save_total_limit=args.save_total_limit,
    num_train_epochs=args.num_train_epochs,
    predict_with_generate=args.predict_with_generate,
    logging_dir=args.logging_dir,
    fp16=args.fp16,
    push_to_hub=args.push_to_hub,
)

# 初始化訓練器
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 開始訓練
trainer.train()
