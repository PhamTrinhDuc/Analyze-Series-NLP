import torch
import os
import gc
import dotenv
import huggingface_hub
import pandas as pd
import numpy as np
import evaluate
import torch.nn as nn
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from transformers import (
    pipeline, 
    Trainer,
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments
)
from datasets import Dataset
from utils.cleaner import Cleaner
from configs.configurator import CONFIGURATOR

dotenv.load_dotenv()


class CustomerTrainer(Trainer):
    """
    Mục đích chính là tùy chỉnh cách tính toán loss function trong quá trình huấn luyện mô hình.
    """
    def compute_loss(self, model, inputs, return_outputs = False):
        labels = inputs.get("labels").float()

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits 

        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights).to(self.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss   
    
    def set_class_weights(self, class_weights):
        self.class_weights = class_weights
    
    def set_device(self, device):
        self.device = device


class JutsuClassifier:
    def __init__(self, 
                 model_name: Optional[str] = CONFIGURATOR.MODEL_NAME_JUTSU,
                 token_df: Optional[str] = os.getenv("HF_TOKEN"),
                 model_path: Optional[str] = None) -> None:
        
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = CONFIGURATOR.JUTJU_PATH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_dict = {0: "Genjutsu", 1: "Ninjutsu", 2: "Taijutsu"}
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        huggingface_hub.login(token=token_df)

    def _simplify_jutsu(self, jutsu: str) -> str:
        for jutsu_type in self.label_dict.values():
            if jutsu_type in jutsu:
                return jutsu_type

    def _load_tokenizer(self) -> AutoTokenizer:
        if huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer
    
    def _load_model(self):
        if self.model_path is None:
            model = pipeline(task="text-classification", 
                         model=self.model_path, 
                         return_all_scores=True, 
                         device=self.device)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                num_labels = len(self.label_dict),
                id2label=self.label_dict,
        )
        return model

    def load_data(self):
        df = pd.read_json(self.data_path, lines=True)
        df['jutsu'] = df['jutsu'].apply(self.simplify_jutsu)
        df['text'] = df['jutsu_name'] + ". " + df['description']
        df = df[['text', 'jutsu']]
        df = df.dropna()

        # clean text
        cleaner = Cleaner()
        df['text_cleaned'] = df['text'].apply(cleaner.clean)

        # encode label
        le = LabelEncoder()
        le.fit(df['jutsu'].tolist())
        df['label'] = le.transform(df['jutsu'].tolist())

        # train test split
        data_train, data_test = train_test_split(df, test_size=0.2, stratify=['label'])

        # convert to huggingface dataset
        data_train = Dataset.from_pandas(data_train)
        data_test = Dataset.from_pandas(data_test)

        #tokenize dataset
        tokenied_train = data_train.map(lambda example: self.tokenizer(example['text_cleaned'], truncation=True, padding='max_length'), batched=True)
        tokenied_test = data_test.map(lambda example: self.tokenizer(example['text_cleaned'], truncation=True, padding='max_length'), batched=True)
        return tokenied_train, tokenied_test
    

    def get_class_weights(self, df: pd.DataFrame):
        class_weights = compute_class_weight(class_weight='balanced', 
                                             classes=sorted(df['label'].unique().tolist()), 
                                             y=df['label'])
        return class_weights
    
    def compute_metrics(self, eval_pred):
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def training(self, train_data, test_data):
        train_data_df = train_data.to_pandas()
        test_data_df = test_data.to_pandas()
        df = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
        class_weights = self.get_class_weights(df=df)

        model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                num_labels = len(self.label_dict),
                id2label=self.label_dict,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            learning_rate=2e-4,
            per_device_eval_batch_size=8,
            per_device_train_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            push_to_hub=True,
        )

        trainer = CustomerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.set_class_weights(class_weghts=class_weights)
        trainer.set_device(device=self.device)

        # Flush memory
        del trainer, model # xóa biến trainer và model khỏi bộ nhớ
        gc.collect() # kích hoạt quá trình thu gom rác, giải phóng bộ nhớ từ các đối tượng không còn được sử dụng.

        if self.device == 'cuda':
            torch.cuda.empty_cache() # giải phóng một số bộ nhớ GPU không được sử dụng bởi PyTorch.

    def postprocess(self, output):
        output = []
        for pred in output:
            label = max(pred, key=lambda x: x['score'])['label']
            output.append(label)
        return output

    def inference(self, text):
        output = self.model(text)
        predictions = self.postprocess(output)
        return predictions