import torch.nn.functional as F
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
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn.functional as F

def focal_loss(outputs, labels, gamma=2.0, alpha=1.0):
    ce_loss = F.cross_entropy(outputs, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
    return focal_loss

def train(model, data_loader, optimizer, criterion, device, lr_scheduler):
    # set the model to train mode
    model.train()

    # initialize the loss, accuracy, precision, recall, and f1_score variables
    total_loss, total_accuracy = 0, 0
    total_precision, total_recall, total_f1 = 0, 0, 0

    # iterate over the data loader
    for data in tqdm(data_loader):
        # move the inputs to the device
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        # zero the gradient
        optimizer.zero_grad()

        # get the model's predictions
        outputs = model(input_ids, attention_mask)

        # get the focal loss
        loss = focal_loss(outputs, labels)
        total_loss += loss.item()

        # accuracy calculation
        logits = outputs.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        total_accuracy += flat_accuracy(logits, label_ids)

        # calculate precision, recall, and f1 score
        total_precision += precision_score(label_ids, np.argmax(logits, axis=1), average='weighted')
        total_recall += recall_score(label_ids, np.argmax(logits, axis=1), average='weighted')
        total_f1 += f1_score(label_ids, np.argmax(logits, axis=1), average='weighted')

        # perform backpropagation and optimization
        loss.backward()
        lr_scheduler.step()
        optimizer.step()

    # calculate the average loss, accuracy, precision, recall, and f1 score
    avg_loss = total_loss / len(data_loader)
    # avg_accuracy = total_accuracy / len(data_loader)
    avg_precision = total_precision / len(data_loader)
    avg_recall = total_recall / len(data_loader)
    avg_f1 = total_f1 / len(data_loader)

    return avg_loss, avg_precision, avg_recall, avg_f1

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def evaluate(model, data_loader, criterion, device):
    # set the model to eval mode
    model.eval()

    # initialize the loss, accuracy, precision, recall, and f1_score variables
    total_loss = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    # iterate over the data loader
    for data in tqdm(data_loader):
        # move the inputs to the device
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        # disable gradient computation
        with torch.no_grad():
            # get the model's predictions
            outputs = model(input_ids, attention_mask)

            # get the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # calculate the number of correct predictions
            logits = outputs.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            total_accuracy += flat_accuracy(logits, label_ids)

            # calculate precision, recall, and f1 score
            total_precision += precision_score(label_ids, np.argmax(logits, axis=1), average='weighted')
            total_recall += recall_score(label_ids, np.argmax(logits, axis=1), average='weighted')
            total_f1 += f1_score(label_ids, np.argmax(logits, axis=1), average='weighted')

    # calculate the average loss, accuracy, precision, recall, and f1 score
    avg_loss = total_loss / len(data_loader)
    # accuracy = total_accuracy / len(data_loader)
    precision = total_precision / len(data_loader)
    recall = total_recall / len(data_loader)
    f1 = total_f1 / len(data_loader)

    return avg_loss, precision, recall, f1

# Function to calculate F1 score, Precision, Recall
def calculate_metrics(predictions, labels):
    predictions = torch.argmax(predictions, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')

    return f1, precision, recall

from sklearn.model_selection import train_test_split
import pandas as pd

def stratified_split(csv_path, stratify_column, test_size=0.2, random_state=2024, debug=False):
    """
    Split data into training and development sets using stratified splitting.
    
    Parameters:
    - csv_path (str): Path to the CSV file.
    - stratify_column (str): Column name to use for stratification.
    - test_size (float): Proportion of data to use for the development set (default = 0.2).
    - random_state (int): Seed for random number generator to ensure reproducibility (default = 2024).
    
    Returns:
    - train_df (pd.DataFrame): Training set.
    - dev_df (pd.DataFrame): Development set.
    """
    # Read data from CSV file
    data = pd.read_csv(csv_path, index_col=0)
    
    # Check if the stratify column exists
    if stratify_column not in data.columns:
        raise ValueError(f"Column '{stratify_column}' does not exist in the dataset")

    if debug: 
        # Check distribution of the classes
        print(f"Number of unique classes: {len(data[stratify_column].value_counts())}")
    
    # Perform stratified split
    train_df, dev_df = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=data[stratify_column]
    )
    
    if debug:
        # Display class distribution in the split sets
        print("\nClass distribution in the training set:")
        print(train_df[stratify_column].value_counts())
        
        print("\nClass distribution in the development set:")
        print(dev_df[stratify_column].value_counts())
    
    return train_df, dev_df

def double_single_count_products(data, product_column):
    """
    Doubles the rows for product types that appear only once in the given column.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame.
        product_column (str): The name of the column containing product types.
    
    Returns:
        pd.DataFrame: The updated DataFrame with doubled rows for single-count products.
    """
    # Identify products with count = 1
    counts = data[product_column].value_counts()
    single_count_products = counts[counts == 1].index

    # Filter rows with these products
    rows_to_duplicate = data[data[product_column].isin(single_count_products)]

    # Concatenate the original DataFrame with the duplicated rows
    result = pd.concat([data, rows_to_duplicate], ignore_index=True)
    
    return result