import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# Define the BERT tokenizer and model
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
from datetime import datetime
import torch
import torch.nn as nn
from tqdm import tqdm

# Function to generate formatted text
import re

def clean_text(input_text):
    # Loại bỏ khoảng trắng thừa ở đầu và cuối
    cleaned_text = input_text.strip()

    # Thay thế nhiều dòng mới hoặc khoảng trắng thừa bằng một khoảng trắng đơn
    cleaned_text = re.sub(r'[\n\r]+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)

    # Loại bỏ các ký tự đặc biệt không cần thiết
    cleaned_text = re.sub(r'\s+/\s+', '/', cleaned_text)

    # Đảm bảo giữ các chi tiết quan trọng bằng cách tách theo dạng key-value
    lines = re.split(r'(\s{2,}|:)', cleaned_text)
    final_text = ' '.join([line.strip() for line in lines if line.strip()])

    return final_text

def count_and_trim_text(cleaned_text, limit=30000):
    # Đếm số lượng ký tự
    if len(cleaned_text) > limit:
        # Cắt bớt nếu vượt quá giới hạn
        print(len(cleaned_text))
        return cleaned_text[:limit]
    return cleaned_text 


def format_input_text(input_text):
    input_text_clean =  clean_text(input_text) 
    input_text_clean_trim = count_and_trim_text(input_text_clean)
    return input_text_clean_trim


# Tokenize and encode the sentences
class MyDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, inference=False):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inference = inference


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # year,month,day,country,title,text

        year = self.data.iloc[index]['year']
        month = self.data.iloc[index]['month']
        day = self.data.iloc[index]['day']
        country = self.data.iloc[index]['country']
        title = self.data.iloc[index]['title']
        text = self.data.iloc[index]['title'] +  " /n " +  self.data.iloc[index]['text'] 
        
        if self.inference == False:         
            label = self.data.iloc[index]['label']
            if "Document" in self.data.columns:
                text = self.data.iloc[index]['Document'] 
            # else: 
            #     print("Don't have Documents --> Create Documents... ")

        # Tạo input_text rõ ràng và ngữ nghĩa
        # input_text = (
        #     f"On {year}-{month:02d}-{day:02d}, in {country}, an event titled '{title}' was reported. "
        #     f"Here is the full context: {text} [SEP]"
        # )
        # input_text = f"On {year}-{month:02d}-{day:02d}, occurred in {country}. Context: {text}"
        input_text = text 

        input_text = format_input_text(input_text)
        #input_text = f"In {year}, '{title}' happened in {country}. Context: {text} [SEP]"
        # input_text = f"{year}[SEP] {title} [SEP] {text} [SEP]"
        # input_text = f"{title} /n {text}"

        # print(input_text)

        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        if self.inference == False:   
            return {
                'text': input_text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else: 
            return {
                'text': input_text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }