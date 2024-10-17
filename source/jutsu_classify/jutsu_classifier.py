import torch
import os
import dotenv
import huggingface_hub
import pandas as pd
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
    def compute_loss():
        pass
    




class JutsuClassifier:
    def __init__(self) -> None:
        self.MODEL_NAME = CONFIGURATOR.MODEL_NAME_JUTSU
        self.data_path = CONFIGURATOR.JUTJU_PATH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_dict = {0: "Genjutsu", 1: "Ninjutsu", 2: "Taijutsu"}
        self.tokenizer = self.load_tokenizer()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.MODEL_NAME,
            num_labels = len(self.label_dict),
            id2label=self.label_dict,
        )
        huggingface_hub.login(token=os.getenv("HF_TOKEN"))


    def simplify_jutsu(self, jutsu: str) -> str:
        for jutsu_type in self.label_dict.values():
            if jutsu_type in jutsu:
                return jutsu_type

    def load_tokenizer(self):
        if huggingface_hub.repo_exists(CONFIGURATOR.MODEL_JUTSU_PATH):
            tokenizer = AutoTokenizer.from_pretrained(CONFIGURATOR.MODEL_JUTSU_PATH)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        return tokenizer
    
    def load_model_exists(self) -> pipeline:
        model = pipeline(task="text-classification", model=CONFIGURATOR.MODEL_JUTSU_PATH, return_all_scores=True, device=self.device)
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

    def training(self, train_data, test_data, class_weghts):
        training_args = TrainingArguments(
            output_dir=CONFIGURATOR.MODEL_JUTSU_PATH,
            learning_rate=2e-4,
            per_device_eval_batch_size=8,
            per_device_train_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            push_to_hub=True,
        )