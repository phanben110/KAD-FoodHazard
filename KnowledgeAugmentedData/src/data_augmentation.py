# data_augmentation.py
import os
from tqdm import tqdm
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from src.evaluation import Evaluation
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.abstract_retrieval.utils import *
from langchain_ollama import OllamaEmbeddings
from src.utils import get_logger

import warnings 
warnings.filterwarnings('ignore')


from src.utils import *
import time
import re

from enum import Enum

class TaskType(Enum):
    SEMEVAL25T9 = "SemEval_2025_task9"

class DataAugmentation:
    def __init__(self, llm, prompt_template, vectorstore=None):
        self.llm = llm
        self.vectorstore = vectorstore
        self.prompt_templates = prompt_template
        self.evaluation = Evaluation(llm, self.prompt_templates.get_evaluation_prompt_template())
        self.local_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=os.getenv("OLLAMA_HOST_EMBEDDING"))
        self.logger = get_logger(__name__) 
    
    def pipeline_1_without_RAG(self, df, batch_size=10, filename="batch_results.json", task = TaskType.BIORED, checker=True ): 
        
        if task == TaskType.SEMEVAL25T9:
            augmentation_prompt_template = self.prompt_templates.get_sem_eval_2025_task9_prompt_template() 

        # Clean the data and extract samples and IDs
        print(f"----------- Start task {task} pipeline 1 without RAG ")
        print(f"----------- Save to: {filename}")
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        samples = list(df["Document"])
        sample_ids = list(df["SampleID"])

        # Splitting samples and sample IDs into batches
        batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
        batch_ids = [sample_ids[i:i + batch_size] for i in range(0, len(sample_ids), batch_size)]

        # Initialize the prompt templates and tags
        prompt = prompt_template(augmentation_prompt_template)
        chain = prompt | self.llm

        # The main loop
        for batch, ids in tqdm(zip(batches, batch_ids), total=len(batches), desc="Processing Batches"):
            inputs = []
            for instance, id in zip(batch, ids):
                inputs.append({"messages": [HumanMessage(content=instance)], "passage": instance})
            if batch_size == 1: 
                batch_responses = chain.invoke(inputs[0])
            else: 
                batch_responses = chain.batch(inputs)

            # Evaluation step
            inputs_val = []
            batch_matching_score = []
            for instance, augmented, id in zip(batch, batch_responses, ids):               
                if task == TaskType.SEMEVAL25T9: 
                    matching_score = 0
                batch_matching_score.append(matching_score)
                if batch_size == 1: 
                    inputs_val.append({'original_text': instance, 'augmented_text': augmented[1]})
                else:
                    inputs_val.append({'original_text': instance, 'augmented_text': augmented}) 

            if batch_size == 1:
                if checker==False:
                    save_single_to_json(batch_responses=batch_responses, 
                                        batch_matching_score=batch_matching_score, 
                                        batch_responses_evaluation=10, 
                                        ids=ids, batch=batch, filename=filename)  
                else:
                    batch_responses_evaluation = self.evaluation.evaluate_single(inputs_val[0])
                    if len(batch_responses_evaluation['text']) > 1: 
                        save_single_to_json(batch_responses, batch_matching_score, batch_responses_evaluation['text'][1], ids, batch, filename) 
            else: 
                if checker==False: 
                    batch_responses_evaluation =  [{"text":10}]*len(inputs_val)
                else:
                    batch_responses_evaluation = self.evaluation.evaluate_batch(inputs_val)
                save_batch_to_json(batch_responses, batch_matching_score, batch_responses_evaluation, ids, batch, filename)


    def pipeline_2_with_RAG(self, df, batch_size=10, filename="batch_results.json", task=TaskType.BIORED):
 
        if task == TaskType.SEMEVAL25T9:
            augmentation_RAG_prompt_template = self.prompt_templates.get_sem_eval_2025_task9_prompt_template_rag()
    
        # Clean the data and extract samples and IDs
        print( "----------- Start pipeline 2 with RAG ")
        print(f"----------- Save to: {filename}")
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        samples = list(df["Document"])
        sample_ids = list(df["SampleID"])

        # Splitting samples and sample IDs into batches
        batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
        batch_ids = [sample_ids[i:i + batch_size] for i in range(0, len(sample_ids), batch_size)]

        # Initialize the prompt templates and tags
        rag_prompt = ChatPromptTemplate.from_template(augmentation_RAG_prompt_template)
        chain = (
            RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

        # The main loop
        for batch, ids in tqdm(zip(batches, batch_ids), total=len(batches), desc="Processing Batches"):
            inputs = []
            for instance, id in zip(batch, ids):
                # Retry mechanism
                attempts = 0
                max_attempts = 3
                while attempts < max_attempts:
                    try:
                        docs = self.vectorstore.similarity_search(instance, k=2) 
                        # # print log length of doc 
                        # print(f"Length of docs: {len(docs.keys())}")
                        break  # Break the loop if no error occurs
                    except Exception as e:
                        attempts += 1
                        if attempts < max_attempts:
                            print(f"Error: {e}. Retrying in 1-2 seconds...")
                            time.sleep(1 + attempts)  # Delay increases with each retry
                        else:
                            print(f"Failed after {max_attempts} attempts.")
                            docs = []  # Fall back to an empty result or handle it in another way
                inputs.append({"context": docs, "passage": instance})

            batch_responses = chain.batch(inputs)

            # Evaluation step
            inputs_val = []
            batch_matching_score = []
            for instance, augmented, id in zip(batch, batch_responses, ids):
                if task == TaskType.SEMEVAL25T9: 
                    matching_score = 25             
                batch_matching_score.append(matching_score)
                inputs_val.append({'original_text': instance, 'augmented_text': augmented})

            batch_responses_evaluation = self.evaluation.evaluate_batch(inputs_val)
            save_batch_to_json(batch_responses, batch_matching_score, batch_responses_evaluation, ids, batch, filename)


    def pipeline_3_RAG_PubMed_API(self, df, batch_size=1, filename="batch_results.json", task=TaskType.BIORED, config=None):    
        from metapub import PubMedFetcher
        from src.abstract_retrieval.PubMedAbstractRetriever import PubMedAbstractRetriever
        from src.abstract_retrieval.LocalJSONStore import LocalJSONStore
        from src.abstract_retrieval.ChromaDbRag import ChromaDbRag

        pubmed_fetcher = PubMedFetcher()
        abstract_retriever = PubMedAbstractRetriever(pubmed_fetcher)
        storage_folder_path = config["pipeline"]["pubmed_retrieval"]["storage_path"]
        store = LocalJSONStore(storage_folder_path)
        persist_directory = "backend/chromadb_storage"
        rag_workflow = ChromaDbRag(persist_directory, self.local_embeddings)

        if task == TaskType.SEMEVAL25T9:
            augmentation_RAG_prompt_template = self.prompt_templates.get_sem_eval_2025_task9_prompt_template_rag()
    
        # Clean the data and extract samples and IDs
        print( "----------- Start pipeline 3 RAG with PubMed API ")
        print(f"----------- Save to: {filename}")
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        samples = list(df["Document"])
        sample_ids = list(df["SampleID"])

        # Splitting samples and sample IDs into batches
        batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
        batch_ids = [sample_ids[i:i + batch_size] for i in range(0, len(sample_ids), batch_size)]

        # Initialize the prompt templates and tags
        rag_prompt = ChatPromptTemplate.from_template(augmentation_RAG_prompt_template)
        chain = (
            RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

        # The main loop
        for batch, ids in tqdm(zip(batches, batch_ids), total=len(batches), desc="Processing Batches"):
            inputs = []
            for instance, id in zip(batch, ids):
                # Retry mechanism
                attempts = 0
                attempts_pubmed = 0
                max_attempts = 3
                max_attempts_pubmed = 3
                while attempts < max_attempts:
                    try:
                        # Step 1: Use PubMedAbstractRetriever to get abstract data for a query "Does abamectin cause cancer?"
                        if task == TaskType.SEMEVAL25T9:
                            clean_instance = clean_text_semeval(instance)
                        abstracts, simple_query  = abstract_retriever.get_abstract_data(self.llm, clean_instance)
                        #print(f"Abstracts: {abstracts}")
                        # Step 2: Use the retrieved data with LocalJSONStorage to persist them in local storage
                        query_id = store.save_dataset(abstracts, clean_instance) 
                        # Step 3: Use ChromDBRAGWorkflow to create a vector index using the list of documents created via LocalJSONStorage 
                        documents = store.read_documents(query_id)

                        if len(documents) == 0:
                            self.logger.error(f"No documents found for the query: {abstracts}")
                            attempts_pubmed += 1
                            if attempts_pubmed < max_attempts_pubmed:
                                print(f"Error: No documents found for the query. Retrying in 1 seconds...")
                                time.sleep(1)
                            elif attempts_pubmed == max_attempts_pubmed:
                                self.logger.error(f"Failed after {max_attempts_pubmed} attempts. Using query local") 
                                vector_all_index = rag_workflow.get_vector_index_by_user_query("all")
                                docs = vector_all_index.similarity_search(simple_query, k=2) 
                                print(f"--------------All: {len(docs)}---------------")
                                break
                                
                            # docs = self.vectorstore.similarity_search(instance, k=2) 
                        else: 
                             
                            vector_index = rag_workflow.create_vector_index_for_user_query(documents, query_id)
                            rag_workflow.create_vector_index_for_user_query(documents, "all")

                            # Run similarity search on newly created index, using the original user query: 
                            docs = vector_index.similarity_search(clean_instance, k=2)
                            #rag_workflow.delete_vector_index_by_user_query(query_id)
                            
                            break  # Break the loop if no error occurs

                    except Exception as e:
                        attempts += 1
                        if attempts < max_attempts:
                            print(f"Error: {e}. Retrying in 1-2 seconds...")
                            time.sleep(1 + attempts)  # Delay increases with each retry
                        else:
                            print(f"Failed after {max_attempts} attempts.")
                            docs = []  # Fall back to an empty result or handle it in another way
                inputs.append({"context": docs, "passage": instance})

            batch_responses = chain.batch(inputs)

            # Evaluation step
            inputs_val = []
            batch_matching_score = []
            for instance, augmented, id in zip(batch, batch_responses, ids):
                if task == TaskType.SEMEVAL25T9: 
                    matching_score = 100

                batch_matching_score.append(matching_score)
                inputs_val.append({'original_text': instance, 'augmented_text': augmented})

            batch_responses_evaluation = self.evaluation.evaluate_batch(inputs_val)
            save_batch_to_json(batch_responses, batch_matching_score, batch_responses_evaluation, ids, batch, filename)

