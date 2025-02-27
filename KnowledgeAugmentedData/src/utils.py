# utils.py

import json
import re
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import logging
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Function to define prompt template
def prompt_template(template):
    prompt = ChatPromptTemplate.from_messages([template, MessagesPlaceholder(variable_name="messages")])
    return prompt 

# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def check_response(response):
    if hasattr(response, 'content'):
        return response.content
    else:
        return response

def replace_placeholder(input_text, input_replace, value_replace):
    """
    Replaces the specified placeholder in the input text with the provided value.

    Parameters:
    input_text (str): The original text containing placeholders.
    input_replace (str): The placeholder to be replaced.
    value_replace (str): The value to replace the placeholder with.

    Returns:
    str: The text with the placeholder replaced by the value.
    """
    return input_text.replace(input_replace, value_replace)

# Function to save the results of each batch to a JSON file
def save_single_to_json(batch_responses, batch_matching_score, batch_responses_evaluation, batch_ids, batch, filename="batch_results.json"):
    batch_data = {}
    for idx, (response, score, evaluation_score, id) in enumerate(zip(batch_responses, batch_matching_score, batch_responses_evaluation, batch_ids)):
        batch_data[id] = {
            "original_text": batch[idx],
            "augmented_text": check_response(response),
            "matching_score": score,
            "evaluation_score": evaluation_score
        }
    
    # Load existing data if the file exists
    try:
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    # Update with new batch data
    existing_data.update(batch_data)

    # Save updated data back to the JSON file
    with open(filename, 'w') as file:
        json.dump(existing_data, file, indent=4) 


def save_to_json_BERT_Pipeline(batch_responses, batch_matching_score, batch_responses_evaluation, batch_ids, batch, filename="batch_results.json"):
    batch_data = {}
    for idx, (response, score, evaluation_score, id) in enumerate(zip(batch_responses, batch_matching_score, batch_responses_evaluation, batch_ids)):
        batch_data[id] = {
            "original_text": batch[idx],
            "augmented_text": response,
            "matching_score": score,
            "evaluation_score": evaluation_score
        }
    
    # Load existing data if the file exists
    try:
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    # Update with new batch data
    existing_data.update(batch_data)

    # Save updated data back to the JSON file
    with open(filename, 'w') as file:
        json.dump(existing_data, file, indent=4) 


# Function to save the results of each batch to a JSON file
def save_batch_to_json(batch_responses, batch_matching_score, batch_responses_evaluation, batch_ids, batch, filename="batch_results.json"):
    batch_data = {}
    for idx, (response, score, evaluation_score, id) in enumerate(zip(batch_responses, batch_matching_score, batch_responses_evaluation, batch_ids)):
        batch_data[id] = {
            "original_text": batch[idx],
            "augmented_text": check_response(response),
            "matching_score": score,
            "evaluation_score": evaluation_score['text']
        }
    
    # Load existing data if the file exists
    try:
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    # Update with new batch data
    existing_data.update(batch_data)

    # Save updated data back to the JSON file
    with open(filename, 'w') as file:
        json.dump(existing_data, file, indent=4)



def get_logger(name, file_name='backend/logs/AugLLms_Log.log', level=logging.INFO):
    """
    Set up a logger that logs messages to both a file and the console.

    Args:
        name (str): Name of the logger, typically the module's `__name__`.
        log_dir (str): Directory where log files will be saved.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create file handler to log to a file
    log_file = file_name
    # print("log_file", log_file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create console handler to log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Change to logging.DEBUG if you want detailed console logs

    # Create formatter for the log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# # Usage
# logger = get_logger(__name__)

# # Log messages
# logger.debug("This is a debug message.")
# logger.info("This is an info message.")
# logger.error("This is an error message.")


# Updated function to split text into sentences
def split_data(text):
    sentences = re.split(r'(?<!\w\.\w\.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+', text)
    return sentences

# Function to check for the presence of "@", "$", and "@/"
def check_symbols(sentence):
    if "@" in sentence and "$" in sentence and "@/" in sentence:
        return 0
    return 1



def balance_data_v4(df, class_column, large_factor=0.4, medium_factor=0.45, small_factor=0.4, cap_ratio=0.25):
    """
    Cân bằng dữ liệu với nhiều lớp, có điều chỉnh theo cấp độ số lượng.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu ban đầu.
        class_column (str): Tên cột class cần cân bằng.
        large_factor (float): Hệ số điều chỉnh cho lớp lớn.
        medium_factor (float): Hệ số điều chỉnh cho lớp trung bình.
        small_factor (float): Hệ số điều chỉnh cho lớp nhỏ.
        cap_ratio (float): Tỷ lệ giới hạn tối đa số lượng được tạo thêm cho lớp nhỏ.

    Returns:
        pd.DataFrame: DataFrame đã được cân bằng.
    """
    # Tính số lượng của từng class
    class_counts = df[class_column].value_counts()
    max_count = class_counts.max()
    
    # Phân chia các lớp thành nhóm theo cấp độ
    tiers = {
        "large": class_counts[class_counts > max_count * 0.5].index,
        "medium": class_counts[(class_counts > max_count * 0.1) & (class_counts <= max_count * 0.5)].index,
        "small": class_counts[class_counts <= max_count * 0.1].index,
    }

    # Tạo danh sách số lượng cần bổ sung
    class_to_add = {}
    for cls, count in class_counts.items():
        if cls in tiers["large"]:
            num_to_add = int((max_count - count) * large_factor)
        elif cls in tiers["medium"]:
            num_to_add = int((max_count - count) * medium_factor)
        else:  # Lớp nhỏ
            num_to_add = int((max_count - count) * small_factor)
            # Áp dụng giới hạn số lượng bổ sung cho lớp nhỏ
            num_to_add = min(num_to_add, int(max_count * cap_ratio))
        class_to_add[cls] = num_to_add

    # DataFrame mới để lưu trữ dữ liệu cân bằng
    balanced_df = df.copy()

    # Bổ sung dữ liệu cho từng class
    for cls, num_to_add in class_to_add.items():
        if num_to_add > 0:
            class_data = df[df[class_column] == cls]
            sampled_data = class_data.sample(n=num_to_add, replace=True, random_state=42)
            balanced_df = pd.concat([balanced_df, sampled_data], ignore_index=True)

    return balanced_df


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

def check_word_count_difference(aug, original):
    # Calculate the word counts
    aug_word_count = len(aug.split())
    original_word_count = len(original.split())
    
    # Determine the larger word count for the denominator
    max_word_count = max(aug_word_count, original_word_count) 
    if max_word_count == 0: 
        return 0
    
    # Calculate the percentage difference
    difference = abs(aug_word_count - original_word_count)
    percentage_difference = (difference / max_word_count) * 100 
    return 100 - percentage_difference  

def calculate_semantic_similarity(text1, text2, model):
    """
    Calculate the semantic similarity score between two texts using Sentence-BERT.
    
    Parameters:
    - text1: First text (string).
    - text2: Second text (string).
    
    Returns:
    - similarity_score: Cosine similarity score (float, 0.0 to 1.0).
    """
    # Load Sentence-BERT model

    
    # Generate embeddings for both texts
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity_score = util.cos_sim(embedding1, embedding2).item()
    # print(similarity_score*100)
    
    return similarity_score*100