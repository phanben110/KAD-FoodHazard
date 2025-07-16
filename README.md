# MyMy at SemEval-2025 Task 9: A Robust Knowledge-Augmented Data Approach for Reliable Food Hazard Detection

**KAD-FoodHazard** is an advanced pipeline developed for **SemEval-2025 Task 9**, focusing on enhancing food hazard detection through Knowledge-Augmented Data (KAD). By integrating Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs) and fine-tuning techniques, this system addresses challenges such as class imbalance and hallucination in food incident reports.

## Pipeline Components

- **KnowledgeAugmentedData**: Augments training data using RAG and domain-specific knowledge.
- **FineTuneFoodHazard**: Fine-tunes models on the enriched dataset to improve classification accuracy.

## Features

- **Retrieval-Augmented Generation (RAG)** utilizing the PubMed API for domain-specific knowledge.
- **LLM-based data augmentation** employing models like Llama 3.1 and Mixtral.
- **Validation filtering** to ensure high-quality augmented data.
- **Fine-tuning** of state-of-the-art models such as PubMedBERT and Gemini Flash.
- **Ensemble strategies** for robust predictions.

## System Architecture

The system follows four key steps:

1. **Information Retrieval**: Retrieves relevant documents from external sources like PubMed.
2. **Data Generation**: Uses LLMs to generate augmented samples based on retrieved knowledge.
3. **Validation Filtering**: Filters low-quality data using a scoring system to ensure dataset integrity.
4. **Fine-Tuning**: Trains models on the enriched dataset and employs ensemble strategies for improved performance.

![System Architecture](https://raw.githubusercontent.com/phanben110/KAD-FoodHazard/refs/heads/master/images/DA_Method_V2.png)

## Installation

### Step 1: Install Python Environments

Clone the repository and set up the required environments:

```bash
# Clone the repository
git clone https://github.com/phanben110/KAD-FoodHazard.git
cd KAD-FoodHazard

# Install KnowledgeAugmentedData environment
conda env create -f augLLMs.yml
```

### Step 2: Install Ollama

Install Ollama, a tool for managing LLMs locally:

```bash
curl -fsSL https://ollama.com/install.sh | sh

# Download required LLMs
ollama pull llama3.1:8b
ollama pull mixtral
```

### Step 3: Install Google Cloud SDK

Follow the official [Google Cloud SDK installation guide](https://cloud.google.com/sdk/docs/install#linux).

## Usage

### 1. KnowledgeAugmentedData

This module generates high-quality augmented data using Retrieval-Augmented Generation (RAG).

**Steps:**

1. Navigate to the `KnowledgeAugmentedData` directory:

   ```bash
   cd KnowledgeAugmentedData
   ```

2. Configure the augmentation settings in `config/augmentation_cfg.yaml`.

3. Run the main script to generate augmented data:

   ```bash
   python main.py --config ./../config/augmentation_cfg.yaml --task SemEval_2025_task9 --aug
   ```

*Output*: Augmented data will be saved in `datasets/SemEval_2025_task9/demo_augmentation_llama3.1:8b`.

### 2. FineTuneFoodHazard

Once the augmented data is ready, fine-tune models for food hazard detection.

**Steps:**

1. Navigate to the `FineTuneFoodHazard` directory:

   ```bash
   cd FineTuneFoodHazard
   ```

2. Update the fine-tuning configuration in `config/finetune_cfg.yaml` (e.g., specify model type, dataset paths).

3. Run the fine-tuning script:

   ```bash
   python main.py --config ./../config/finetune_cfg.yaml
   ```

## Demo

💻 **Watch our system in action**: [YouTube Demo](https://www.youtube.com/watch?v=lWsnHEVbFig)

## Authors

- **[Ben Phan](https://phanben110.github.io/)**  
  Master in Computer Science at National Cheng Kung University.

- **[Jung-Hsien Chiang](https://www.csie.ncku.edu.tw/en/members/18)**  
  Distinguished Professor at the Department of Computer Science and Information Engineering, NCKU.

