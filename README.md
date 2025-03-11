# 📋 项目简介

这是一个基于BERT模型的文本分类API服务，使用FastAPI构建。该服务可以对输入文本进行分类，返回预测标签和置信度分数。BERT (Bidirectional Encoder Representations from Transformers) 是一种强大的预训练语言模型，在多种NLP任务中表现出色。

## ✨ 功能特点

* 🚀 基于FastAPI的高性能API服务
* 🧠 集成预训练BERT模型进行文本分类
* 📊 返回详细的分类结果和置信度分数
* 📝 支持自定义模型和分类标签
* 📚 提供交互式API文档
* 🔄 支持热重载，便于开发

## 🛠️ 技术栈

* **FastAPI** : 现代、高性能的Web框架
* **Transformers** : 提供BERT等预训练模型的库
* **PyTorch** : 深度学习框架
* **Uvicorn** : ASGI服务器

## 📦 安装步骤

### 前置条件

* Python 3.8 或更高版本
* pip 包管理器

### 安装指南

1. **克隆仓库 todo**

```bash
git clone https://github.com/huxiaolongyin/HTWBertClassifier.git
cd HTWBertClassifier
```

2. **安装 uv**(一个高效的package管理)

```bash
pip install uv
```

3. **创建并激活虚拟环境**

```bash
uv venv --python <指定版本>
```

在Windows上:

```bash
.venv\Scripts\activate
```

在Linux/Mac上:

```bash
source .venv/bin/activate
```

3. **安装依赖**

```bash
uv pip install -r requirements.txt
```

## 🚀 使用方法

### 启动服务

```bash
uvicorn api.main:app --port 6565 --host 0.0.0.0
```

或者使用一键启动

```bash
# Linux/Mac
./start.sh

# Windows
./start.bat
```

服务将在本地的6565端口启动。

### API端点

* **GET /** - 健康检查
* **POST /classify** - 文本分类接口

### API文档

启动服务后，访问以下URL查看交互式API文档:

* [http://localhost:6565/docs](command:_cody.vscode.open?%22http%3A%2F%2Flocalhost%3A8000%2Fdocs%22) - Swagger UI
* [http://localhost:6565/redoc](command:_cody.vscode.open?%22http%3A%2F%2Flocalhost%3A8000%2Fredoc%22) - ReDoc

## 📊 API示例

### 请求示例

```bash
curl -X 'POST' \
  'http://localhost:6565/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "今天天气怎么样"
}'
```

### 响应示例

```json
{
  "text": "今天天气怎么样",
  "label": "天气",
  "score": 0.9845,
}
```

## 🧠 模型说明

默认使用基于 bert-chinese 预训练模型进行多分类任务，您可以通过修改 `model.py` 文件中的配置来:

* 使用不同的预训练模型
* 调整分类类别数量
* 自定义标签名称
* 加载您自己已微调的模型

### 自定义模型(todo)

修改 `model.py` 文件中的 `HTWBertClassifier` 类初始化参数:

```python
# 使用自定义模型
classifier = HTWBertClassifier(model_name="./your_fine_tuned_model")

# 自定义标签
# self.id2label = {
#     0: "category_1", 
#     1: "category_2", 
#     2: "category_3"
# }
```

## 🔧 项目结构

```bash
project/
├── api/
│   ├── __init__.py         # 包初始化文件
│   ├── main.py             # FastAPI应用入口
│   ├── model.py            # BERT模型封装
│   └── schemas.py          # Pydantic数据模型
├── model/
│   ├── htw_bert_text_cls/  # 模型文件
│   └── model_turning/      # 模型微调
├── requirements.txt        # 项目依赖
└── README.md               # 项目文档
```

## 📈 性能优化

* 首次加载模型可能需要几秒钟时间
* 对于生产环境，建议:
  * 使用更小的模型（如DistilBERT）提高推理速度
  * 启用模型量化减少内存占用
  * 使用GPU加速（如果可用）
  * 配置适当的工作进程数量
