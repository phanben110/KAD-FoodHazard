import configparser
import os
import pandas as pd
import warnings
from langchain_openai import AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAI

# Ignore all warnings
warnings.filterwarnings('ignore')
class LLMs:
    """
    Class to create the appropriate LLM based on the configuration file.
    """

    LLAMA3_8B = "llama3.1:8b"
    LLAMA3_70B = "llama3.1:70b"
    GPT_35_TURBO = "gpt-35-turbo"
    GPT_4 = "gpt-4o-mini-2000"
    MIXTRAL = "mixtral"
    GEMINI = "gemini-2.0-flash"


    def __init__(self, config):
        self.config = config
        self.temperature = 0.7 
        self.LLAMA3_8B = "llama3.1:8b"
        self.LLAMA3_80B = "llama3.1:70b"
        self.GPT_35_TURBO = "gpt-35-turbo"
        self.GPT_4 = "gpt-4o-mini-2000"
        self.MIXTRAL = "mixtral"
        self.GEMINI = "gemini-2.0-flash"
        self.llm_name = None

    def create_llm(self, llm_name = "llama3.1:8b"):
        """
        Create and return an LLM based on the config file.
        """
        if llm_name == self.LLAMA3_8B :
            llm = ChatOllama(model="llama3.1:8b", temperature=self.temperature ) 
            self.llm_name = self.LLAMA3_8B
            return llm

        elif llm_name == self.LLAMA3_70B :
            llm = ChatOllama(model=self.LLAMA3_70B, temperature=self.temperature ) 
            self.llm_name = self.LLAMA3_70B
            return llm 

        elif llm_name == self.GPT_4:
            llm = AzureChatOpenAI(
            azure_endpoint=self.config["azure_openai"].get("AZURE_ENDPOINT"),
            azure_deployment="gpt-4o-mini-2000",
            openai_api_version="2024-02-15-preview",
            model='gpt-4o-mini-2000', 
            temperature=self.temperature 
            ) 
            self.llm_name = self.GPT_4
            return llm 
        
        elif llm_name == self.GPT_35_TURBO: 
            llm = AzureChatOpenAI(
            azure_endpoint=self.config["azure_openai"].get("AZURE_ENDPOINT"),
            azure_deployment="gpt-35-turbo",
            openai_api_version="2024-05-01-preview",
            model='gpt-35-turbo', 
            temperature=self.temperature 
            )
            self.llm_name = self.GPT_35_TURBO
            return llm
        
        elif llm_name == self.MIXTRAL: 
            llm = ChatOllama( model="mixtral", temperature=self.temperature  ) 
            self.llm_name = self.MIXTRAL
            return llm

        elif llm_name == self.GEMINI: 
            # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            llm = VertexAI(model_name="gemini-1.5-flash-001", project_id="bens-thesis-augllms")
            self.llm_name = self.GEMINI 
            return llm 

        else:
            raise ValueError("No valid LLM configuration found.")