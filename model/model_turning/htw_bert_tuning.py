import os
import pickle
import platform
from typing import Union
import pandas as pd
import torch
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
import tqdm
from transformers import (
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification,
)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from config import set_env

# 创建udf时，重新加载该
set_env()


class HTWBertModel:
    """
    汉特云公司使用Bert微调模型的代码

    用于文本分类
    """

    # 加载配置
    # 加载 .env 文件中的环境变量
    model_path = os.getenv("OUTPUT_MODEL_PATH")
    bert_model_path = os.getenv("BERT_MODEL_PATH")
    learning_rate = float(os.getenv("LEARNING_RATE"))
    epochs = int(os.getenv("EPOCHS"))

    def __init__(self):
        """
        加载训练好的 Bert 模型，加载 tokenizer

        Args:
            bert_model_path: Bert 模型路径
        """
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        with open(self.model_path + "/label_encoder.pkl", "rb") as f:
            self.loladed_le = pickle.load(f)

    def __base_model(self):
        """
        加载基础的 Bert 模型，加载 tokenizer
        """
        self.model = BertForSequenceClassification.from_pretrained(
            self.bert_model_path,
            num_labels=3,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)

    def tuning(self, train_dataset, valid_dataset, batch_size):
        """
        使用 transformers 的 AdamW 优化器和 linear schedule 进行微调
        """
        # 加载基础的 Bert 模型，加载 tokenizer, 加载参数
        self.__base_model()

        # 数据加载器
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False
        )

        # 如果 GPU 可用，则将模型移动到 GPU 上
        if torch.cuda.is_available():
            self.model.cuda()
            device = "cuda"
        else:
            device = "cpu"

        total_steps = len(train_dataloader) * self.epochs
        # 优化器
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        # 学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        for epoch in range(self.epochs):
            # 训练模式
            self.model.train()
            train_loss, valid_loss = 0, 0
            eval_accuracy = 0

            # 训练进度条
            train_pbar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Epoch {epoch+1}/{self.epochs}",
            )

            for batch in train_pbar:
                # 梯度清零
                self.model.zero_grad()
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[3].to(device)
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs[0]
                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # 评估模式
            self.model.eval()
            valid_pbar = tqdm(
                valid_dataloader,
                total=len(valid_dataloader),
                desc=f"Epoch {epoch+1}/{epochs}",
            )
            for batch in valid_pbar:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[3].to(device)
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs[0]
                valid_loss += loss.item()
                logits = outputs[1]
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to("cpu").numpy()
                eval_accuracy += self.flat_accuracy(logits, label_ids)

            print(f"Train Epoch: {epoch+1}")
            print(f"Training Loss: {train_loss/len(train_dataloader):.3f}")
            print(f"Validation Loss: {valid_loss/len(valid_dataloader):.3f}")
            print(f"Training Accuracy: {eval_accuracy/len(valid_dataloader):.3f}")
            print("\n")

    def save_model(self, model_path):
        """
        使用 transformers 的 save_pretrained 方法，保存模型和tokenizer
        """
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    @staticmethod
    def __batch(iterable, batch_size: int = 1):
        """进行分批处理"""
        length = len(iterable)
        for idx in range(0, length, batch_size):
            yield iterable[idx : min(idx + batch_size, length)]

    @staticmethod
    def __get_model_path():
        """判断使用平台，获取模型路径， 加载模型"""
        root_path = (
            "D:/code/datawarehouse/"
            if platform.system() == "Windows"
            else "/root/datawarehouse/"
        )
        return os.path.join(root_path, "src/model/htw_bert_text_cls")

    def predict(self, data: Union[str, list], batch_size=16):
        """调用Bert微调模型, 进行文本分类"""
        if data == "" or data is None:
            return "无"

        from torch import argmax

        data = [data] if isinstance(data, str) else data

        if isinstance(data, pd.DataFrame):
            pass

        def get_prediction(data):
            """获取预测结果"""
            input = self.tokenizer(
                data,
                truncation=True,
                padding=True,
                max_length=50,
                return_tensors="pt",
            )
            output = self.model(**input)
            preds = argmax(output.logits, dim=-1).tolist()
            return self.loladed_le.inverse_transform(preds)

        if (len(data)) == 1:
            return get_prediction(data)[0]

        predicitons = []
        for batch_data in self.__batch(data, batch_size):
            predicitons.extend(get_prediction(batch_data))

        return predicitons


class HTWDataLoader:
    """
    负责对输入的 dataframe 文本数据进行预处理(__preprocess_data), 使用模型对文本内容进行分词(tokenize)处理，然后返回一个model可训练的 dataloader 数据

    Args:
        data: pd.DataFrame 数据
        batch_size: int 批处理大小
        text_col: int 文本列索引
        label_col: int 标签列索引
        shuffle: bool 是否打乱数据

    Example:
        >>> data = pd.DataFrame({"text": ["hello world", "hello world"], "label": ["1", "2"]})
        >>> data_loader = DataLoader(data, batch_size=16, text_col=0, label_col=1, shuffle=False)
    """

    model_path = os.getenv("BERT_MODEL_PATH")
    label_count_least = int(os.getenv("LABEL_COUNT_LEAST"))
    random_seed = int(os.getenv("RANDOM_SEED"))
    max_length = int(os.getenv("MAX_LENGTH"))
    batch_size = int(os.getenv("BATCH_SIZE"))

    def __init__(self, data, text_col=0, label_col=1, shuffle=False):
        self.data_processed = self.__preprocess_data(
            data,
            text_col,
            label_col,
            label_count_least=self.label_count_least,
            random_seed=42,
        )

        if test_size == 0:
            return self._to_dataloader(batch_size, max_length, data)
        else:
            train_data, test_data = train_test_split(
                data, test_size=test_size, random_state=random_seed
            )
            train_dataloder = self._to_dataloader(batch_size, max_length, train_data)
            test_dataloder = self._to_dataloader(batch_size, max_length, test_data)
            return train_dataloder, test_dataloder

    def __preprocess_data(
        self,
        data: pd.DataFrame,
        text_col: int,
        label_col: int = None,
        label_count_least: int = None,
        random_seed: int = None,
    ):
        """
        对 dataframe 数据进行预处理，包含去重，标准化等

        Args:
            data: pd.DataFrame 数据
            text_col: int 文本列索引
            label_col: int 标签列索引
            label_count_least: int 标签最少数
            random_seed: int 随机种子

        Returns:
            pd.DataFrame 处理后的数据
        """
        temp_df = pd.DataFrame()
        # 进行去重，提高数据质量
        data = data.drop_duplicates().dropna()

        # 选择文本列
        temp_df["text"] = data.iloc[:, text_col]

        # 如果有标签列，则认为是训练数据，对标签进行序列化编码
        if label_col:
            temp_df["label"] = data.iloc[:, label_col]

            # 选择标签有一定数量的数据，去除标签数量过少的标签
            label_value_count = temp_df["label"].value_counts()
            label_list = list(
                label_value_count[label_value_count >= label_count_least].index
            )
            temp_df = temp_df[temp_df["label"].isin(label_list)]

            # 唤醒词只有一个值，增加当前标签最少数 min_label_count 条唤醒词数据
            min_label_count = temp_df["label"].value_counts().min()
            wake_df = pd.DataFrame({"text": ["笨笨同学"], "label": ["唤醒词"]})
            wake_df = pd.concat([wake_df] * min_label_count, ignore_index=True)
            temp_df = pd.concat([temp_df, wake_df], ignore_index=True)

            # 对标签进行序列化编码
            label_encoder = LabelEncoder()
            temp_df["label"] = label_encoder.fit_transform(temp_df["label"])

            # 进行数据均等分，避免标签数据量不均衡，导致模型对某个标签权重更高
            result_data = pd.DataFrame()
            for label in temp_df["label"].unique():
                df_sampled = temp_df[temp_df["label"] == label].sample(
                    n=min_label_count, random_state=random_seed
                )
                result_data = pd.concat([result_data, df_sampled])

        # 没有标签列，则认为是预测数据，直接选择文本列
        else:
            result_data = data.iloc[:, text_col]

        return result_data

    def _tokenize(
        self,
        data,
    ):
        """
        使用 BertTokenizer 对文本进行分词(tokenize)处理

        如果是dataframe ， 进行分词处理， 返回 tokenized_ids, tokenized_mask, tokenized_type_ids

        Args:
            data: pd.DataFrame 数据或者是字符串
            max_length: int 最大长度
        """
        if isinstance(data, pd.DataFrame):
            text = list(data["text"].values)
            label = list(data["label"].values)
        else:
            text = list(data)

        tokenizer = BertTokenizer.from_pretrained(self.model_path)
        tokenized_sentence = tokenizer(
            text,
            return_tensors="pt",  # 返回pytorch tensor类型
            max_length=self.max_length,  # 最大长度
            padding="max_length",  # 填充长度
            truncation=True,
        )

        tokenized_ids = tokenized_sentence["input_ids"]
        tokenized_mask = tokenized_sentence["attention_mask"]
        tokenized_type_ids = tokenized_sentence["token_type_ids"]

        if isinstance(data, pd.DataFrame):
            return (
                tokenized_ids,
                tokenized_mask,
                tokenized_type_ids,
                tensor(label).long(),
            )
        else:
            return tokenized_ids, tokenized_mask, tokenized_type_ids

    def _to_dataloader(self, data):
        """将 dataframe 转换为 dataloader"""
        tensors = self._tokenize(self.max_length, data)
        dataset = TensorDataset(*tensors)
        return DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data["text"][index]
        label = self.data["label"][index]
        return self._tokenize(text, label)

    def get_dataloader(
        self,
        data: pd.DataFrame,
        text_loc: int = 0,
        label_loc: int = 1,
        label_count_least: int = 150,
        random_seed: int = 42,
        batch_size: int = 5,
        test_size: float = 0,
        max_length: int = 100,
    ):
        """
        param:
            data: 输入的数据
            text_loc: 文本列的位置
            label_loc: 标签列的位置
            label_count_least: 标签最少数量
            random_seed: 随机种子
            batch_size: 批次大小
            test_size: 测试集比例, 测试集比例不为0, 则返回训练集和测试集
            max_length: padding的最大长度
        return: dataloader数据
        """
        data = self._preprocess_data(
            data, text_loc, label_loc, label_count_least, random_seed
        )
        if test_size == 0:
            return self._to_dataloader(batch_size, max_length, data)
        else:
            train_data, test_data = train_test_split(
                data, test_size=test_size, random_state=random_seed
            )
            train_dataloder = self._to_dataloader(batch_size, max_length, train_data)
            test_dataloder = self._to_dataloader(batch_size, max_length, test_data)
            return train_dataloder, test_dataloder

    def get_label_count(self):
        """获取标签数量"""
        return self.label_encoder.classes_.shape[0]

    def save_label_encoder(self, path: str = "output/label_encoder.pkl"):
        """保存标签映射文件"""
        with open(path, "wb") as f:
            pickle.dump(self.label_encoder, f)
