import spacy
import os
from ast import literal_eval
from nltk import sent_tokenize
from utils.data_loader import load_subtiles_dataset
from configs.configurator import CONFIGURATOR

class NamedEntityRecognizer:
    
    def __init__(self):
        self.df = load_subtiles_dataset()
        self.model = self.load_model()

    def load_model(self):
        # !python -m spacy download en_core_web_trf
        nlp_model = spacy.load('en_core_web_trf')
        return nlp_model
    
    def get_named_entities(self, scripts):
        """
        Trích xuất các thực thể được đặt tên (named entities) từ danh sách các kịch bản.
        Hàm này nhận vào một danh sách các kịch bản, sau đó sử dụng mô hình nhận diện thực thể 
        để tìm các thực thể có nhãn "PERSON". Các tên được trích xuất sẽ được lưu trữ dưới dạng 
        tập hợp các tên riêng (first name) và trả về dưới dạng danh sách các tập hợp.
        Args:
            scripts (list): Danh sách các kịch bản dưới dạng chuỗi.
        Returns:
            list: Danh sách các tập hợp chứa các tên riêng được trích xuất từ các kịch bản.
        """
        ner_output = set()
        script_sentences = sent_tokenize(scripts)
        for sample_script in script_sentences:
            sentence_ner = self.model(sample_script)
            ners = set()
            for entity in sentence_ner.ents:
                if entity.label_ == "PERSON":
                    full_name = entity.text.strip()
                    first_name = full_name.split(" ")[0].strip()
                    if first_name.lower() not in ['the', 'that', 'this', 'those'] and len(first_name) > 1:
                        ners.add(first_name)

            if len(ners) > 0:
                ner_output.update(list(ners))
        print(list(ner_output))
        return list(ner_output)
    
    def get_ners(self):
        """
        Lấy các thực thể được đặt tên (Named Entities) từ dữ liệu kịch bản.
        Phương thức này kiểm tra xem tệp tin lưu trữ thực thể đã tồn tại hay chưa.
        Nếu tệp tin tồn tại, nó sẽ đọc dữ liệu từ tệp tin và chuyển đổi các thực thể từ chuỗi sang đối tượng Python.
        Nếu tệp tin không tồn tại, nó sẽ thực hiện nhận diện thực thể từ dữ liệu kịch bản và lưu kết quả vào tệp tin.
        Returns:
            pandas.DataFrame: DataFrame chứa các thực thể được đặt tên.
        """

        # if os.path.exists(CONFIGURATOR.SAVE_THEME_SCRIPT_PATH) and 'ners' in self.df.columns:
        #     self.df['ners'] = self.df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
        #     return self.df

        self.df['ners'] = self.df['scripts'].apply(self.get_named_entities)
        self.df.to_csv(CONFIGURATOR.SAVE_THEME_SCRIPT_PATH, index=False)
        return self.df
    
    def testing(self):
        df  = self.get_ners()
        print(df.iloc[0]['ners'])
