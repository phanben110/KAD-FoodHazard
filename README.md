# **MyMy at SemEval-2025 Task 9: A Robust Knowledge-Augmented Data**
Approach for Reliable Food Hazard Detection**

## **Overview**
KAD-FoodHazard is a robust pipeline designed for **SemEval-2025 Task 9** to improve food hazard detection using **Knowledge-Augmented Data (KAD)**. By integrating **Retrieval-Augmented Generation (RAG)** with advanced **Large Language Models (LLMs)** and fine-tuning techniques, this system addresses challenges such as class imbalance and hallucination in food incident reports.

The pipeline consists of two main components:
1. **KnowledgeAugmentedData**: Augments training data using RAG and domain-specific knowledge.
2. **FineTuneFoodHazard**: Fine-tunes models on the enriched dataset to achieve high accuracy in food hazard classification.

---

## **Features**
- Retrieval-Augmented Generation (RAG) with PubMed API for domain-specific knowledge.
- LLM-based data augmentation using models like Llama 3.1 and Mixtral.
- Validation filtering to ensure high-quality augmented data.
- Fine-tuning of state-of-the-art models such as PubMedBERT and Gemini Flash.
- Ensemble strategies for robust predictions.

---

## **System Architecture**
The system comprises four key steps:
1. **Information Retrieval**: Simplifies queries and retrieves relevant documents from external sources like PubMed.
2. **Data Generation**: Uses LLMs to generate augmented samples based on retrieved knowledge.
3. **Validation Filtering**: Filters low-quality data using a scoring system to ensure dataset integrity.
4. **Fine-Tuning**: Trains models on the enriched dataset and combines predictions using an ensemble strategy.

### **System Workflow Diagram**
![System Architecture]()

---

## **Installation**

### Step 1: Install Python Environments
Clone the repository and set up the required environments:

```bash
# Clone the repository
git clone https://github.com/phanben110/KAD-FoodHazard.git
cd KAD-FoodHazard

# Install KnowledgeAugmentedData environment
conda env create -f augLLMs.yml

# Install FineTuneFoodHazard environment
conda env create -f biored.yml
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

---

## **Usage**

### **1. KnowledgeAugmentedData**
This module generates high-quality augmented data using Retrieval-Augmented Generation (RAG).

#### Steps:
1. Navigate to the `KnowledgeAugmentedData` directory:

```bash
cd KnowledgeAugmentedData
```

2. Configure the augmentation settings in `config/augmentation_cfg.yaml`.

3. Run the main script to generate augmented data:

```bash
python main.py --config ./../config/augmentation_cfg.yaml --task SemEval_2025_task9 --aug
```

The augmented data will be saved in `datasets/SemEval_2025_task9/demo_augmentation_llama3.1:8b`.

---

### **2. FineTuneFoodHazard**
Once the augmented data is ready, fine-tune models for food hazard detection.

#### Steps:
1. Navigate to the `FineTuneFoodHazard` directory:

```bash
cd FineTuneFoodHazard
```

2. Update the fine-tuning configuration in `config/finetune_cfg.yaml` (e.g., specify model type, dataset paths).

3. Run the fine-tuning script:

```bash
python main.py --config ./../config/finetune_cfg.yaml
```

---

## **Authors**
- **Ben Phan**, NCKU
- **Jung-Hsien Chiang**, NCKU

