# 代码修改记录

本文档记录了从原始项目代码开始，为在 **Python 3.14 + Mac M 芯片**环境下成功运行所做的全部修改。

---

## 修改 1：`requirements.txt` — 更新依赖版本

### 原因
原始文件指定 `torch==2.0.1`，但 Python 3.14 太新，PyPI 上 torch 最低只提供到 2.9.0 的安装包，导致 `pip install -r requirements.txt` 直接报错：
```
ERROR: Could not find a version that satisfies the requirement torch==2.0.1
```

### 修改内容

| 位置 | 修改前 | 修改后 |
|------|--------|--------|
| 第 1 行 | `# Python 3.9` | `# Python 3.14` |
| 第 2 行 | `torch==2.0.1` | `torch>=2.9.0` |
| 第 3 行 | `torchvision==0.15.2` | 删除 |
| 第 4 行 | `torchaudio==2.0.2` | 删除 |
| 第 5 行 | `torchtext==0.15.2` | 删除 |
| 第 6–10 行 | 各包严格版本号 `==x.y.z` | 改为 `>=x.y.z` |

`torchvision`、`torchaudio`、`torchtext` 均被删除，原因是项目代码中没有任何地方 import 这三个库，属于无用依赖。`torchtext` 在新版 torch 后也已停止维护。

---

## 修改 2：`config.py` — 修正数据集名称

### 原因
新版 `datasets` 库（HuggingFace）要求数据集名称必须包含命名空间，格式为 `namespace/name`。原来的 `"opus_books"` 会报错：
```
HfUriError: Repository id must be 'namespace/name', got 'opus_books'.
```

### 修改位置
`config.py` 第 11 行，`get_config()` 函数内：

```python
# 修改前
"datasource": "opus_books",

# 修改后
"datasource": "Helsinki-NLP/opus_books",
```

---

## 修改 3：`config.py` — 修复文件夹命名中的路径分隔符

### 原因
`datasource` 的值 `"Helsinki-NLP/opus_books"` 中含有 `/`，被用来拼接权重文件夹路径时，`/` 会被操作系统解释为路径分隔符，导致意外创建嵌套目录 `Helsinki-NLP/opus_books_weights/` 而不是预期的单层目录。

### 修改位置
`config.py` 第 23 行，`get_weights_file_path()` 函数：

```python
# 修改前
model_folder = f"{config['datasource']}_{config['model_folder']}"

# 修改后
model_folder = f"{config['datasource'].replace('/', '_')}_{config['model_folder']}"
```

`config.py` 第 29 行，`latest_weights_file_path()` 函数，同样修改。

---

## 修改 4：`train_wb.py` — 修复文件夹命名中的路径分隔符

### 原因
同修改 3，`train_wb.py` 中也有一处直接拼接 datasource 作为目录名的代码。

### 修改位置
`train_wb.py` 第 170 行，`train_model()` 函数内：

```python
# 修改前
Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

# 修改后
Path(f"{config['datasource'].replace('/', '_')}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
```

---

## 修改 5：`train_wb.py` — 替换 HuggingFace 数据集加载方式

### 原因
HuggingFace 对 `Helsinki-NLP/opus_books` 数据集要求登录认证，下载时返回 `403 Forbidden`：
```
Cannot access content at: https://huggingface.co/datasets/Helsinki-NLP/opus_books/...
Make sure your token has the correct permissions.
```
为避免注册 HuggingFace 账号，改为直接从 [OPUS 官网](https://object.pouta.csc.fi) 下载数据，该网站完全公开，无需任何账号。

### 修改位置 1 — import 部分（第 1–19 行）

```python
# 修改前
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split

# 修改后（新增 random, io, zipfile, requests；移除 load_dataset 和 random_split）
import random
import io
import zipfile
import requests
from torch.utils.data import DataLoader
```

### 修改位置 2 — 新增 `download_opus_books()` 函数（在 Tokenizer 部分之前）

新增函数，负责从 OPUS 下载并解析平行语料：

```python
def download_opus_books(lang_src, lang_tgt, cache_dir="opus_data"):
    # 下载到本地缓存，已存在则跳过
    # 解压 zip，读取两个平行文本文件
    # 返回格式：[{"translation": {"en": "...", "it": "..."}}, ...]
```

返回值格式与原来 `load_dataset` 返回的格式完全一致，后续代码无需改动。

### 修改位置 3 — `get_ds()` 函数（约第 150 行）

```python
# 修改前
ds_raw = load_dataset(config["datasource"], f"{config['lang_src']}-{config['lang_tgt']}", split="train")
train_size = int(0.9 * len(ds_raw))
val_size   = len(ds_raw) - train_size
train_raw, val_raw = random_split(ds_raw, [train_size, val_size])

# 修改后
ds_raw = download_opus_books(config["lang_src"], config["lang_tgt"])
random.shuffle(ds_raw)
train_size = int(0.9 * len(ds_raw))
train_raw  = ds_raw[:train_size]
val_raw    = ds_raw[train_size:]
```

`random_split`（PyTorch 工具）替换为普通的列表 shuffle + 切片，效果相同。

---

## 修改 6：`train_wb.py` — 启用 Mac M 芯片 GPU（MPS）

### 原因
原代码只检测 CUDA（NVIDIA 显卡），在 Mac 上永远回退到 CPU，训练极慢（预计 90 小时）。Mac M 系列芯片有内置 GPU，PyTorch 通过 `mps` 后端支持它，速度比 CPU 快 5–10 倍。

### 修改位置 1 — `train_model()` 函数内 device 检测（约第 195 行）

```python
# 修改前
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 修改后
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

### 修改位置 2 — `torch.cuda.empty_cache()` 调用（约第 236 行）

```python
# 修改前
torch.cuda.empty_cache()

# 修改后
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

`torch.cuda.empty_cache()` 是 CUDA 专用 API，在 MPS 设备上调用会报错，因此加上条件判断，只在有 NVIDIA GPU 时调用。

---

## 修改汇总

| 文件 | 修改次数 | 主要原因 |
|------|----------|----------|
| `requirements.txt` | 1 | Python 3.14 不兼容旧版本号 |
| `config.py` | 2 | 数据集命名规范变更；路径分隔符问题 |
| `train_wb.py` | 3 | 路径分隔符；HuggingFace 认证；MPS 支持 |
