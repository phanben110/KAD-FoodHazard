import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import wandb
from src.MyDataset import MyDataset
from src.BertSentimentClassifier import BertSentimentClassifier
from src.utils import train, evaluate, double_single_count_products
import warnings
import numpy as np 
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler, DataCollatorWithPadding

warnings.filterwarnings("ignore")


class FoodHazardDetectionSemEval2025:
    def __init__(self, config, use_wandb = True):
        self.config = config
        self.use_wandb = use_wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and initialize model paths
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_name"])

        # Generate paths for saving models using os.path.join
        self.final_model_path = os.path.join(
            config["out_model_path"],
            f"final_{config['wandb_run_name']}_{config['max_len']}.pt"
        )

        self.best_model_path = os.path.join(
            config["out_model_path"],
            f"best_{config['wandb_run_name']}_{config['max_len']}.pt"
        )

        # Initialize WandB
        name = config["wandb_run_name"] + "_" + str(config['max_len']) 
        print("name:", name)
        if self.use_wandb: 
            self.wandb = wandb.init(
                project=config["wandb_project_name"],
                name=name,
                config=config,
            )

        # Create logs folder
        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"


    def load_data(self):
        # Load and stratified split
        data = pd.read_csv(self.config["data_path"], index_col=0)
        data = double_single_count_products(data, self.config["label_column"])
        train_df, val_df = train_test_split(
            data,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"],
            stratify=data[self.config["label_column"]],
        )

        # Encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(data[self.config["label_column"]])
        train_df["label"] = label_encoder.transform(train_df[self.config["label_column"]])
        val_df["label"] = label_encoder.transform(val_df[self.config["label_column"]])

        self.num_classes = len(label_encoder.classes_) 
        print("Number of class:",label_encoder.classes_)

        label_path = os.path.join(
            self.config["out_model_path"],
            f"{self.config['label_column']}_label_encoder.npy"
        )

        np.save(label_path, label_encoder.classes_) 

        # Create datasets and data loaders
        self.train_loader = DataLoader(
            MyDataset(train_df, self.tokenizer, self.config["max_len"]),
            batch_size=self.config["batch_size"],
            shuffle=True,
        )
        self.val_loader = DataLoader(
            MyDataset(val_df, self.tokenizer, self.config["max_len"]),
            batch_size=self.config["batch_size"],
        )

    def initialize_model(self):
        self.model = BertSentimentClassifier(
            self.config["bert_model_name"], self.num_classes
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        self.criterion = nn.CrossEntropyLoss()

        num_training_steps = self.config["epochs"] * len(self.train_loader)

        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
    # def initialize_model(self):
    #     self.model = BertSentimentClassifier(
    #         self.config["bert_model_name"], self.num_classes
    #     ).to(self.device)
    #     self.optimizer = torch.optim.Adam(
    #         self.model.parameters(), lr=self.config["learning_rate"]
    #     )
    #     self.criterion = nn.CrossEntropyLoss()
        
    #     # Add learning rate scheduler
    #     self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)


    def train_and_evaluate(self):
        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_epoch = 0

        with open(self.log_filename, "a") as log_file:
            for epoch in range(self.config["epochs"]):
                start_time = datetime.now()

                print(f"Epoch {epoch + 1}/{self.config['epochs']}")

                # Training
                train_loss, train_precision, train_recall, train_f1 = train(
                    self.model, self.train_loader, self.optimizer, self.criterion, self.device,  self.lr_scheduler
                )
                train_losses.append(train_loss)

                # Validation
                val_loss, val_precision, val_recall, val_f1 = evaluate(
                    self.model, self.val_loader, self.criterion, self.device
                )
                val_losses.append(val_loss)

                # Log metrics to WandB
                if self.use_wandb: 
                    self.wandb.log({
                        "Train Loss": train_loss, "Train Precision": train_precision,
                        "Train Recall": train_recall, "Train F1": train_f1,
                        "Val Loss": val_loss, "Val Precision": val_precision,
                        "Val Recall": val_recall, "Val F1": val_f1,
                        "Learning Rate": self.optimizer.param_groups[0]["lr"],  # Log current LR
                    })

                print(f"Train -- Loss: {train_loss:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
                print(f"Val   -- Loss: {val_loss:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

                log_file.write(f"Epoch {epoch + 1}/{self.config['epochs']}\n")
                log_file.write(f"Train -- Loss: {train_loss:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}\n")
                log_file.write(f"Val   -- Loss: {val_loss:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}\n")
                log_file.flush()

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), self.best_model_path)

                # # Step the scheduler
                # self.scheduler.step()

                # Time tracking
                elapsed_time = datetime.now() - start_time
                remaining_time = (self.config["epochs"] - epoch - 1) * elapsed_time
                print(f"Time Elapsed: {elapsed_time}, Time Remaining: {remaining_time} \n")

        # Save final model
        torch.save(self.model.state_dict(), self.final_model_path)
        print(f"Best Model: Epoch {best_epoch}, Best Validation Loss: {best_val_loss:.4f}")
        print(f"Path to log file: {self.log_filename}")
        # wandb.finish()
        if self.use_wandb:
            self.wandb.finish() 


    def load_label_encoder(self, label_encoder_path):
        """Load the label encoder."""
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
        return label_encoder


    def load_bert_model(self, weight_path, bert_model_name, num_classes, device):
        """Load the BERT model with pre-trained weights."""
        model = BertSentimentClassifier(bert_model_name, num_classes)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        return model.to(device)


    def prepare_test_loader(self, test_data_path, tokenizer, max_len, batch_size):
        """Prepare the DataLoader for test data."""
        test_df = pd.read_csv(test_data_path, index_col=0)
        test_dataset = MyDataset(test_df, tokenizer, max_len, inference=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader


    def predict_labels_and_probs(self, model, test_loader, label_encoder, device):
        """Perform inference and return predicted labels and probabilities."""
        model.eval()
        predicted_labels = []
        predicted_probs = []

        with torch.no_grad():  # Disable gradient computation
            for data in tqdm(test_loader, desc="Predicting"):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)

                # Get model outputs
                outputs = model(input_ids, attention_mask)

                # Predicted labels and probabilities
                preds = torch.argmax(outputs, dim=1).tolist()
                probs = torch.nn.functional.softmax(outputs, dim=1)[0].tolist()

                predicted_labels.extend(label_encoder.inverse_transform(preds))
                predicted_probs.append(probs)

        return predicted_labels, predicted_probs


    def inference(self, test_data_path, label_encoder_path, weight_path, bert_model_name, max_len=200, batch_size=1, device="cuda"):
        """Main function to run the inference."""
        # Load resources
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        label_encoder = self.load_label_encoder(label_encoder_path)
        num_classes = len(label_encoder.classes_)
        model = self.load_bert_model(weight_path, bert_model_name, num_classes, device)

        # Prepare test data
        test_loader = self.prepare_test_loader(test_data_path, tokenizer, max_len, batch_size)

        # Perform inference
        predicted_labels, predicted_probs = self.predict_labels_and_probs(model, test_loader, label_encoder, device)

        return predicted_labels, predicted_probs



if __name__ == "__main__":
    config = {
        "data_path": "data/incidents_train.csv",
        "label_column": "hazard-category",
        "bert_model_name": "alvaroalon2/biobert_diseases_ner",
        "max_len": 50,
        "batch_size": 32,
        "epochs": 5,
        "learning_rate": 2e-5,
        "test_size": 0.2,
        "random_state": 42,
        "final_model_path": "models/hazard_category_final.pt",
        "best_model_path": "models/hazard_category_best.pt",
        "wandb_project_name": "SemEval2025",
        "wandb_run_name": "ST1_hazard_category",
    }

    detection_pipeline = FoodHazardDetectionSemEval2025(config)
    detection_pipeline.load_data()
    detection_pipeline.initialize_model()
    detection_pipeline.train_and_evaluate()
