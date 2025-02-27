import yaml
import os
from src.FoodHazardDetectionSemEval2025 import FoodHazardDetectionSemEval2025

def load_config(config_path):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_pretrained_model(trainer, pretrained_model_path):
    """
    Load a pretrained model from the specified path if it exists.
    """
    if os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}...")
        trainer.load_model(pretrained_model_path)
    else:
        print(f"No pretrained model found at {pretrained_model_path}. Starting training from scratch.")

def train_task(task, subtasks, config):
    """
    Train specified subtasks for a given task.
    """
    task_config = config['tasks'].get(task, {})
    if not task_config:
        raise ValueError(f"Task '{task}' not found in configuration.")

    # Handle 'all' subtasks
    if subtasks == "all":
        subtasks = task_config.keys()

    for subtask in subtasks:
        subtask_config = task_config.get(subtask, {})
        if not subtask_config:
            raise ValueError(f"Subtask '{subtask}' not found under task '{task}' in configuration.")

        # Merge common and subtask-specific configurations
        merged_config = {**config['common'], **subtask_config} 
        print(merged_config)

        # Initialize model trainer
        print(f"Initializing model for task '{task}' and subtask '{subtask}'...")
        detection_pipeline = FoodHazardDetectionSemEval2025(merged_config, use_wandb=True)
        detection_pipeline.load_data()
        detection_pipeline.initialize_model()
        print(f"Training model for task '{task}' and subtask '{subtask}'...")
        detection_pipeline.train_and_evaluate()

def create_folder(): 
    import os

    # List of folder paths to check or create
    folders = [
        'models/ST2/hazard', 
        'models/ST2/product', 
        'models/ST1/hazard_category', 
        'models/ST1/product_category'
    ]

    # Iterate through the folder paths
    for folder in folders:
        # Check if the folder exists
        if not os.path.exists(folder):
            # If the folder doesn't exist, create it
            os.makedirs(folder, exist_ok=True)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")

def main():
    # Load configuration
    create_folder()
    config_path = "config/config.yaml"
    config = load_config(config_path)

    # Run tasks based on the 'run' section
    for task_entry in config.get('run', []):
        task = task_entry.get('task')
        subtasks = task_entry.get('subtasks', "all")
        print(f"Train task: {task} , subtask: {subtasks}")
        train_task(task, subtasks, config) 


if __name__ == "__main__":
    main()
