# chinese-to-tailo-neural-machine-translation-NYCU312707024

## 專案概述

本專案旨在訓練一個用於翻譯的序列到序列（Seq2Seq）機器學習模型，將中文翻譯為台灣南部語言Tailo。此模型基於 Hugging Face Transformers 框架，使用 BART 模型進行微調。專案涵蓋數據預處理、模型訓練和性能評估等多個階段，並採用 Mean Levenshtein Distance 作為評估標準。

BART（Bidirectional and Auto-Regressive Transformers）模型是一種創新的自然語言處理技術，由Facebook於2019年提出。這種模型結合了BERT（Bidirectional Encoder Representations from Transformers）的雙向編碼器架構和GPT（Generative Pre-trained Transformer）的自回歸解碼器架構，使其在自然語言生成、翻譯和理解等多種任務上表現出色。

BART特別擅長於文本生成任務，它通過隨機打亂句子順序或使用遮罩符號替換文本中的某些部分來進行預訓練。這種獨特的“去噪”訓練策略使BART能夠有效地理解和重構語言信息，從而在後續的微調階段能更好地適應特定的語言任務。

選擇BART作為這項任務的模型主要基於其優異的序列到序列處理能力，使其非常適合於機器翻譯任務。BART不僅能理解源語言的深層語義，還能有效生成目標語言的流暢文本。此外，由於其出色的文本理解和生成能力，BART在多種自然語言處理任務中都取得了新的最佳效果，特別是在涉及深度理解和複雜語言生成的場景中。

綜上所述，BART的這些特點使其成為執行中文到Tailo語言翻譯這一特定任務的理想選擇。它的強大功能能夠幫助提高翻譯的準確性和自然性，從而更有效地進行跨語言溝通。


## 安裝需求

建議使用 Python 3.10.13。

在運行本專案之前，需要安裝以下 Python 库：

- torch、torchvision、torchaudio（使用特定的 PyTorch 版本，可從 `https://download.pytorch.org/whl/cu118` 獲取）
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tqdm
- csv
- transformers
- argparse
- datasets
- evaluate
- json

可以使用以下命令來安裝這些套件：
```bash
pip install -r requirement.txt
```

這些套件將為您的專案提供必要的運行和數據處理功能，包括深度學習模型訓練、數據處理和可視化等。



## 資料取得方式

中文台羅平行語料庫的訓練集和測試集可以從 Kaggle 網站上的一個特定競賽 - "Machine Learning 2023 NYCU Translation" 中獲得，在執行時必須使用完整的資料集，而不是使用`example_train.csv`。這些數據集包括以下幾部分：

1. **訓練集** - 包含中文文句及其對應的台羅文翻譯：
   - `train-ZH.csv`: 包含中文文句的訓練數據。
   - `train-TL.csv`: 包含相對應的台羅文翻譯的訓練數據。

2. **測試集** - 用於評估翻譯模型性能的中文文句：
   - `test-ZH-nospace.csv`: 用於模型測試的中文文句，不含空格。

3. **翻譯結果範例** - 展示預期翻譯輸出的格式：
   - `submission.csv`: 台羅文翻譯結果的範例格式。

您可以通過以下網址訪問並下載這些數據集：[Kaggle - Machine Learning 2023 NYCU Translation](https://www.kaggle.com/competitions/machine-learning-2023nycu-translation/data)。



## 文件結構

- `data_proc.py`: 數據處理腳本，負責讀取和預處理數據。
- `train.py`: 主腳本，用來加载數據、訓練模型並生成模型檔案。
- `test.py`: 用於模型的測試和評估。



## `data_proc.py`

### 功能描述
`data_proc.py` 是一個用於數據預處理的腳本，專門處理翻譯任務的文本數據。這個腳本的主要功能是讀取原始的 CSV 格式數據文件，並將其轉換為適合於序列到序列（Seq2Seq）模型訓練的格式。它處理包括中文到 Tailo 語言的文本數據，並將其轉換為模型可以直接使用的 JSON 格式。

### 使用說明

1. 確保已安裝 pandas 和 transformers 套件。如果未安裝，可以使用以下命令安裝：
   ```bash
   pip install pandas transformers
   ```

2. 確保您的 CSV 數據文件格式正確，並包含需要翻譯的文本。

3. 使用命令行參數執行腳本：
   ```bash
   python data_proc.py --input your_input_file.csv --output preprocessed_data.json
   ```

### 參數說明

- `--input`: 指定原始 CSV 格式數據文件的路徑。這應該包含翻譯的源文本和目標文本。
- `--output`: 處理後數據的輸出 JSON 文件的路徑。

### 腳本輸出

腳本運行完成後，將會生成一個 JSON 格式的文件，其中包含了用於訓練的數據。這個 JSON 文件將直接被用於後續的模型訓練過程。

### 注意事項

- 請確保輸入的 CSV 文件格式正確，包括必要的欄位和分隔符。
- 轉換過程可能會根據數據大小和複雜度而花費不同的時間。
- 請確保您的數據集包含足夠的樣本，以提高模型的泛化能力。


## `train.py` 

### 功能描述
`train.py` 是一個用於訓練機器翻譯模型的 Python 腳本。這個腳本的主要功能是讀取和處理翻譯任務的訓練數據集，設定模型參數，並進行模型的訓練過程。腳本中使用了深度學習框架（如 PyTorch）和 Hugging Face 的 Transformers 庫，這些工具提供了強大的功能來支持模型的訓練和微調。

### 主要功能

- **數據加載和預處理**：腳本會加載訓練數據集，如 `train-ZH.csv` 和 `train-TL.csv`，並將其轉換成適合模型訓練的格式。
- **模型設定**：根據所選用的預訓練模型（如 BART），腳本將設定相應的參數，包括學習率、批次大小等。
- **訓練過程**：進行模型的訓練，包括前向傳播、損失計算和反向傳播。
- **模型保存**：訓練完成後，模型將被保存至指定位置，以便後續的使用或進一步的微調。

### 使用方法

1. 確保已安裝所需的依賴，如 PyTorch 和 Transformers。
2. 將訓練數據集放置於指定的文件夾中。
3. 通過命令行執行腳本，並根據需要設定參數，如模型保存路徑和訓練參數。

   ```bash
   python train.py --input_dir ./data --output_dir ./model_output
   ```

### 需要的命令行參數

- `--input_dir`：包含訓練數據的目錄路徑。
- `--output_dir`：模型輸出和保存的目錄路徑。

### 注意事項

- 請根據您的硬件配置適當調整訓練參數，如批次大小和學習率，以確保訓練過程的穩定性。
- 監控訓練過程中的損失和性能指標，以評估模型的訓練效果。
- 確保訓練數據的質量和多樣性，以提高模型的泛化能力。

### Loss log
| Epoch | Training Loss | Validation Loss |
|------:|--------------:|----------------:|
|     1 |        No log |        0.873129 |
|     2 |      0.330100 |        0.394787 |
|     3 |      0.330100 |        0.347354 |
|     4 |      0.166700 |        0.418194 |
|     5 |      0.112100 |        0.316831 |
|     6 |      0.112100 |        0.316632 |
|     7 |      0.075800 |        0.365969 |
|     8 |      0.050500 |        0.501023 |
|     9 |      0.050500 |        0.494580 |
|    10 |      0.034800 |        0.463611 |

## `test.py` 

### 功能描述
`test.py` 是一個用於生成翻譯結果並保存到 CSV 文件的 Python 腳本。此腳本主要用於將測試數據集中的中文文句利用預先訓練好的模型進行翻譯，並將翻譯結果以特定格式輸出。

### 主要流程

1. **初始化和參數設置**：腳本首先創建一個命令行參數解析器，用於接收訓練好的模型檢查點、輸入文件和輸出文件的路徑。

2. **模型和 Tokenizer 加載**：腳本使用 `AutoTokenizer` 和 `AutoModelForSeq2SeqLM` 從提供的檢查點加載 tokenizer 和模型。

3. **數據加載與處理**：使用 pandas 加載測試數據集，並將每行文本轉換為模型可處理的格式。

4. **進行翻譯並保存結果**：腳本逐行對測試數據進行翻譯，並將每個文句的翻譯結果以及其對應的 ID 寫入到指定的 CSV 文件中。

### 使用方法

1. 確保所需的依賴已安裝並且模型檢查點可用。
2. 通過命令行執行腳本，並指定模型檢查點、輸入測試數據和輸出文件的路徑。

   ```bash
   python test.py --checkpoint [model_checkpoint_path] --input [input_csv_path] --output [output_csv_path]
   ```

### 命令行參數

- `--checkpoint`: 訓練好的模型檢查點路徑。
- `--input`: 測試數據集的 CSV 文件路徑。
- `--output`: 要保存翻譯結果的 CSV 文件路徑。

### 注意事項

- 確保指定的模型檢查點與腳本中使用的模型類型相匹配。
- 測試數據集應該符合腳本處理的格式要求。
- 根據需要調整模型的生成參數，如 `max_new_tokens`、`do_sample`、`top_k` 和 `top_p`。
- 確保輸出的翻譯結果與指定的格式相匹配。

### 最佳成績
| Private Score | Public Score |
|:-------------:|:------------:|
|     5.6726    |    5.60416   |

## 作業心得

在這次的翻譯模型訓練和測試專案中，我深入體驗了機器學習和自然語言處理的複雜性與挑戰。從數據預處理到模型訓練，再到最終的測試和性能評估，每個步驟都需要細心的設計和調整。透過這個過程，我不僅提升了對 BART 模型的理解，還加深了對整個機器學習工作流程的認識。

### 學習到的新技術
- **深度學習框架的應用**：在本專案中，我學會了如何使用 PyTorch 和 Transformers 進行深度學習模型的設計和訓練。
- **數據預處理技巧**：我學會了如何有效地處理和準備語料數據，以使其適用於翻譯模型的訓練。
- **BART 模型的微調**：我獲得了實踐經驗，了解如何對 BART 進行微調，以適應特定的翻譯任務。

### 改進方向
- **數據擴充和多樣性**：未來可以通過增加更多樣化的訓練數據來提升模型的泛化能力，這對於提高翻譯質量非常重要。
- **超參數優化**：我認為還有空間進一步優化模型的超參數，如學習率和批次大小，以達到更佳的性能。
- **錯誤分析和模型調整**：深入分析模型在特定案例中的錯誤，並根據這些分析調整模型架構或訓練策略，可能會帶來性能的提升。