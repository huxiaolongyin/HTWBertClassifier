import pickle
from typing import List, Union

import torch
from transformers import BertForSequenceClassification, BertTokenizer


class HTWBertClassifier:
    """
    公司内部使用 Bert-chinese 进行微调的模型，用于客户问询的标签分类
    """

    def __init__(self, model_path="model/htw_bert_text_cls"):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path).to(
                self.device
            )
            self.model.eval()
        except Exception as e:
            print(e)
            raise Exception("加载模型失败")
        try:
            with open(model_path + "/label_encoder.pkl", "rb") as f:
                self.loladed_le = pickle.load(f)
        except Exception as e:
            print(e)
            raise Exception("加载标签编码器失败")

    def predict(self, text: Union[str, List[str]]):
        """
        对数据进行预测
        """
        try:
            # 检查是否为批量输入
            is_batch = isinstance(text, list)
            texts = text if is_batch else [text]

            # 对文本进行编码
            encoded_input = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=50,
                return_tensors="pt",
            )

            # 将输入移动到设备上（GPU或CPU）
            input_ids = encoded_input["input_ids"].to(self.device)
            attention_mask = encoded_input["attention_mask"].to(self.device)

            # 获取模型预测结果
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            # 获取所有样本的预测结果
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)

            results = []
            for i in range(len(texts)):
                predicted_class = predicted_classes[i].item()
                predicted_score = probabilities[i][predicted_class].item()
                results.append(
                    {
                        "text": texts[i],
                        "label": self.loladed_le.inverse_transform([predicted_class])[
                            0
                        ],
                        "score": predicted_score,
                    }
                )

            return results

        except Exception as e:
            print(f"Error: {e}")
            raise Exception("预测失败")


# 创建模型实例（全局单例）
classifier = HTWBertClassifier()

if __name__ == "__main__":
    # 测试代码
    data = "你能做啥"
    result = classifier.predict(data)
    print(result)
