import os
import platform
import pickle
from typing import Union


class HTWTextCls:
    """
    文本分类
    """

    def __init__(self):
        # 加载比较慢，所以只在实例化时，加载库
        from transformers import BertTokenizer, BertForSequenceClassification

        model_path = self.__get_model_path()
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        with open(model_path + "/label_encoder.pkl", "rb") as f:
            self.loladed_le = pickle.load(f)

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

    def htw_text_cls(self, data: Union[str, list], batch_size=16):
        """调用Bert微调模型, 进行文本分类"""
        if data == "" or data is None:
            return "无"

        from torch import argmax

        data = [data] if isinstance(data, str) else data

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
