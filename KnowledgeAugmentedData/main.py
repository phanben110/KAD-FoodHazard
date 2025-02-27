import configparser
import os
import pandas as pd
import warnings
import glob
import argparse
from enum import Enum
from src.data_augmentation import DataAugmentation
from src.template_prompt import PromptTemplates
from src.document_loader import DocumentLoader
from src.retrieval import Retrieval
from src.config import ConfigLoader 
from src.llms import LLMs
from src.data_augmentation import TaskType
from src.utils import *
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from src.merger_data.merger_data_BioRed import MergerDataBioRed
from src.merger_data.merger_data_SemEval_2013_task9 import MergerDataSemEval_2013_task9
from src.merger_data.merger_data_SemEval_2025_task9 import MergerDataSemEval_2025_task9
from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.document_loader import DocumentLoader_2
from src.abstract_retrieval.ChromaDbRag import ChromaDbRag 
from langchain_ollama import OllamaEmbeddings

# Ignore all warnings
warnings.filterwarnings('ignore')
logger = get_logger(__name__) 


def load_config(file_path='config.yaml'):
    # Load configuration
    config_loader = ConfigLoader(file_path)
    return config_loader.load_config()
    

def run_augmentation(task, config, llm, prompt_templates, vectorstore):
    output_path = config["pipeline"]["tasks"][task.value]["output_path"]
    batch_size = config["pipeline"]["tasks"][task.value]["batch_size"]
    pipeline_type = config["pipeline"]["tasks"][task.value]["pipeline_type"]
    
    # Read and preprocess data based on the task type
    if task == TaskType.BIORED:
        # Read the TSV file and add the column names
        column_names = ["PMID", "Type1", "Type2", "Identifier1", "Identifier2", "BoolCls", "LabelCls", "Document", "Relation", "Novelty"]
        df = pd.read_csv(config["pipeline"]["tasks"]["BioRed"]["data_path"], sep='\t', header=None, names=column_names)
        df['SampleID'] = df['PMID'].astype(str) + "_" + df.index.astype(str)
        df = df.dropna(subset=['Relation', 'Novelty'])

    elif task == TaskType.SEMEVAL13T9:
        df = pd.read_csv(config["pipeline"]["tasks"]["SemEval_2013_task9"]["data_path"])
        df = df.reset_index().rename(columns={'index': 'SampleID'})
        df = df[df['pair type'] != 'false']
        df['Document'] = df.apply(add_tag_entities, axis=1) 

    elif task == TaskType.SEMEVAL25T9:
        df = pd.read_csv(config["pipeline"]["tasks"]["SemEval_2025_task9"]["data_path"])
        df = df.reset_index().rename(columns={'index': 'SampleID'})
        # ST1  
        name_column_hazard = "hazard-category"
        name_column_product = "product-category"
        balanced_df_product = balance_data_v4(df, name_column_product, large_factor=0.5, medium_factor=0.55, small_factor=0.5, cap_ratio=0.35)
        df = balance_data_v4(balanced_df_product, name_column_hazard, large_factor=0.5, medium_factor=0.55, small_factor=0.5, cap_ratio=0.35)        

        # ST2 

        # name_column_hazard = "product"
        # name_column_product = "hazard"
        # balanced_df_product = balance_data_v4(df, name_column_product, large_factor=0.2, medium_factor=0.25, small_factor=0.2, cap_ratio=0.15)
        # df = balance_data_v4(balanced_df_product, name_column_hazard, large_factor=0.2, medium_factor=0.25, small_factor=0.2, cap_ratio=0.15) 
        
        # df["Document"] = format_input_text(df['title'] + " /n " + df['text']) 

        df["Document"] = (df['title'] + "\n" + df['text']).apply(format_input_text)
    # Initialize DataAugmentation class
    data_augmentation = DataAugmentation(llm, prompt_templates, vectorstore)
    llm_name = config["pipeline"]["llm"]["models"][config["pipeline"]["llm"]["model"]]

    for i in range(201, 203):
        if pipeline_type <5: 
            print(f"Augmentation {i}: Pipeline: {pipeline_type}, llm: {llm_name}, task: {task.value}, batch size: {batch_size}")
            logger.info(f"Augmentation {i}: Pipeline: {pipeline_type}, llm: {llm_name}, task: {task.value}, batch size: {batch_size}")
        else:
            print(f"Augmentation {i}: Pipeline: {pipeline_type}, task: {task.value}, batch size: {batch_size}")
            logger.info(f"Augmentation {i}: Pipeline: {pipeline_type}, task: {task.value}, batch size: {batch_size}") 

        
        # Use `match` statement for the pipeline type
        match pipeline_type:
            case 1:
                data_augmentation.pipeline_1_without_RAG(
                    df, batch_size=batch_size,
                    filename=f"{output_path}/{task.value}/augmentation_{llm_name}/augmentation_{llm_name}_{i}.json",
                    task=task, checker=True) 
                # if Checker == False --> skip evaluation step
            
            case 2:
                data_augmentation.pipeline_2_with_RAG(
                    df, batch_size=batch_size,
                    filename=f"{output_path}/{task.value}/augmentation_RAG_{llm_name}/augmentation_RAG_{llm_name}_{i}.json",
                    task=task)
            
            case 3:
                data_augmentation.pipeline_3_RAG_PubMed_API(
                    df, batch_size=batch_size,
                    filename=f"{output_path}/{task.value}/augmentation_RAG_API_pubmed_{llm_name}/augmentation_RAG_API_pubmed_{llm_name}_{i}.json",
                    task=task, config=config)
            
            case 4:
                result = data_augmentation.pipeline_4_RAG_PubMed_API_Format_RAGChecker(
                    df, batch_size=batch_size,
                    filename=f"{output_path}/{task.value}/augmentation_RAG_API_pubmed_Format_RAGChecker_{llm_name}/augmentation_RAG_API_pubmed_Format_RAGChecker_{llm_name}_{i}.json",
                    task=task, config=config)
                return result
            
            case 5:
                result = data_augmentation.pipeline_5_Aug_ABEX(
                    df, batch_size=batch_size,
                    filename=f"{output_path}/{task.value}/augmentation_pipeline_5_Aug_ABEX/augmentation_pipeline_5_Aug_ABEX_{i}.json",
                    task=task)
            
            case 6:
                model_path = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
                # model_path = 'cambridgeltl/BioRedditBERT-uncased'
                result = data_augmentation.pipeline_6_Aug_BERT(
                    df, model_path=model_path,
                    batch_size=batch_size,
                    filename=f"{output_path}/{task.value}/augmentation_pipeline_6_Aug_BERT/augmentation_pipeline_6_Aug_BERT_{i}.json",
                    task=task)
            
            case _:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")


def run_RAGChecker(config): 
    # initialize ragresults from json/dict
    input_file = config["pipeline"]["RAGChecker"]["input_data"]
    output_file = config["pipeline"]["RAGChecker"]["output_result"]
    output_folder = os.path.dirname(output_file)  # Extract the folder path from the output file
    llm_name = config["pipeline"]["llm"]["models"][config["pipeline"]["llm"]["model"]]
    host = config["env"]["ollama"]["OLLAMA_HOST"]

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder '{output_folder}' created.")

    with open(input_file) as fp:
        rag_results = RAGResults.from_json(fp.read()) 
    
    # set-up the evaluator
    evaluator = RAGChecker(
        extractor_name=f"ollama/{llm_name}",
        checker_name=f"ollama/{llm_name}",
        checker_api_base=f"http://{host}",
        extractor_api_base=f"http://{host}",
        batch_size_extractor=1,
        batch_size_checker=1
    ) 
    result = evaluator.evaluate(rag_results, all_metrics, output_file)
    
    # Print the result and completion message
    print(result)
    print("RAGChecker evaluation completed successfully.")
    print(f"Save result in path {output_folder}")



def run_merger(task, config):
    # Define the paths for merging
    llm_name = config["pipeline"]["llm"]["models"][config["pipeline"]["llm"]["model"]]
    original_data_file = config["pipeline"]["tasks"][task.value]["merger_config"]["original_data_file"]
    dir_augmentation_data = config["pipeline"]["tasks"][task.value]["merger_config"]["dir_augmentation_data"]
    output_save_path = config["pipeline"]["tasks"][task.value]["merger_config"]["path_output_save"] 
    evaluation_threshold = config["pipeline"]["tasks"][task.value]["merger_config"]["evaluation_threshold"] 
    matching_threshold = config["pipeline"]["tasks"][task.value]["merger_config"]["matching_threshold"] 
    auto_correct = config["pipeline"]["tasks"][task.value]["merger_config"]["auto_correct"]


    print(original_data_file)


    # Use glob to list all JSON files in the directory
    json_files = glob.glob(os.path.join(dir_augmentation_data, "*.json"))

    print(f"Start merging: llm: {llm_name}, task: {task.value}, matching threshold {matching_threshold}, evaluation threshold: {evaluation_threshold}, auto correct: {auto_correct}.")
    logger.info(f"Start merging: llm: {llm_name}, task: {task.value}, matching threshold {matching_threshold}, evaluation threshold: {evaluation_threshold}, auto correct: {auto_correct}.")

    # Instantiate the MergerData class based on the task
    if task == TaskType.BIORED:
        merger = MergerDataBioRed(original_data_file, 
                                  json_files, 
                                  matching_threshold=matching_threshold, 
                                  evaluation_threshold=evaluation_threshold, 
                                  auto_correct=auto_correct)
        merger.process()
        merger.save_to_tsv(output_save_path)

    elif task == TaskType.SEMEVAL13T9:
        merger = MergerDataSemEval_2013_task9(original_data_file, 
                                   json_files, 
                                   matching_threshold=matching_threshold, 
                                   evaluation_threshold=evaluation_threshold, 
                                   auto_correct=auto_correct)
        merger.process()
        merger.save_to_csv(output_save_path)
    # Merger data for SemEval 2025 task9
    elif task == TaskType.SEMEVAL25T9:
        merger = MergerDataSemEval_2025_task9(original_data_file, 
                                   json_files, 
                                   matching_threshold=matching_threshold, 
                                   evaluation_threshold=evaluation_threshold, 
                                   auto_correct=auto_correct)

        print(matching_threshold) 
        print(evaluation_threshold)

        if llm_name == "mixtral":
            print("Apply function process eval v2 using check_word_count_difference and calculate_semantic_similarity.")
            merger.process_eval_v2()
        else: 
            merger.process()
        merger.save_to_csv(output_save_path)

    print("Merging completed.")
    logger.info("Merging completed.")

def run_add_data_to_vector_database(config): 
    print("----------Start add new data in to vector database----------")
    path_database = config["pipeline"]["database"]["path"] 
    embedding_host = config["env"]["ollama"]["OLLAMA_HOST_EMBEDDING"]
    collection_name = config["pipeline"]["database"]["collection_name"]
    data_list = config["pipeline"]["database"]["dataset_list"]

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=embedding_host) 

    database = ChromaDbRag(persist_directory=path_database, 
                       embeddings=embeddings) 
    loader = DocumentLoader_2(data_list)
    data = loader.load() 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(data)  
    database.create_vector_index_for_user_query(all_splits, collection_name)
    database.create_vector_index_for_user_query(all_splits, "all")

    print("---------- Add data finished----------")


def main():
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('--config', required=True, help='Path config file')
    parser.add_argument('--task', required=True, help='Task type BioRed or SemEval_2013_task9 or SemEval_2025_task9')
    parser.add_argument('--merger', default=False, action='store_true', help='Merger data')
    parser.add_argument('--aug', default=False, action='store_true', help='Augmentation data')
    parser.add_argument('--ragchecker', default=False, action='store_true', help='Apply RARChecker to evaluate pipeline')
    parser.add_argument('--add_data', default=False, action='store_true', help='Add data to vector database')
    args = parser.parse_args()

    if args.config:
        global config
        config = load_config(args.config)  
        llm_name = config["pipeline"]["llm"]["models"][config["pipeline"]["llm"]["model"]]

        # Initialize the LLM Factory with config
        llms = LLMs(config)
        llm = llms.create_llm(llm_name)

        # Initialize the prompt templates
        prompt_templates = PromptTemplates()

        # Load the retrieval database
        retrieval_system = Retrieval(
            datasetList=config["pipeline"]["database"]["dataset_list"],
            path_database=config["pipeline"]["database"]["path"], 
            create_database=False
        )
        vectorstore = retrieval_system.load_retrieval()

        if args.task == "BioRed":
            task = TaskType.BIORED
        
        elif args.task == "SemEval_2013_task9":
            task = TaskType.SEMEVAL13T9 
        
        elif args.task == "SemEval_2025_task9":
            task = TaskType.SEMEVAL25T9
        
        if args.aug:
            run_augmentation(task, config, llm, prompt_templates, vectorstore)

        if args.merger:
            run_merger(task, config)  

        if args.ragchecker:
            run_RAGChecker(config)  

        if args.add_data: 
            run_add_data_to_vector_database(config)
            

if __name__ == "__main__":
    main()
