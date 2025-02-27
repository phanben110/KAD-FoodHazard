import configparser
import os
import warnings 
import yaml

from dotenv import load_dotenv

# Ignore all warnings
warnings.filterwarnings('ignore')

class ConfigLoader:
    """
    Class to load configuration from an INI file and set environment variables.
    """
    def __init__(self, file_path='config.ini'):
        self.file_path = file_path
        self.config = self.read_yaml_config()
    
    def read_yaml_config(self): 
        with open(self.file_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def load_config(self):
        """
        Load configuration from a specified ini file and set environment variables.
        """
        
        if 'langchain' in self.config['env']:
            os.environ["LANGCHAIN_TRACING_V2"] = self.config["env"]["langchain"].get("LANGCHAIN_TRACING_V2", "false")
            os.environ["LANGCHAIN_API_KEY"] = self.config["env"]["langchain"].get("LANGCHAIN_API_KEY", "")
            os.environ["LANGCHAIN_PROJECT"] = self.config["env"]["langchain"].get("LANGCHAIN_PROJECT", "")

        if 'ollama' in self.config['env']: 
            os.environ["OLLAMA_HOST"] = self.config["env"]["ollama"].get("OLLAMA_HOST", "")
            os.environ["OLLAMA_HOST_EMBEDDING"] = self.config["env"]["ollama"].get("OLLAMA_HOST_EMBEDDING")
            os.environ["OLLAMA_NUM_PARALLEL"] = self.config["env"]["ollama"].get("OLLAMA_NUM_PARALLEL", "")
            os.environ["OLLAMA_DEBUG"] = self.config["env"]["ollama"].get("OLLAMA_DEBUG")

        if 'azure_openai' in self.config['env']: 
            os.environ["AZURE_OPENAI_API_KEY"] = self.config["env"]["azure_openai"].get("AZURE_OPENAI_API_KEY", "")

        if 'mistral' in self.config['env']: 
            os.environ["MISTRAL_API_KEY"] = self.config["env"]["mistral"].get("MISTRAL_API_KEY", "")

        if 'google_genai' in self.config['env']: 
            os.environ["GOOGLE_API_KEY"] = self.config["env"]["google_genai"].get("GOOGLE_API_KEY", "")

        if 'ncbi_api' in self.config['env']: 
            os.environ["NCBI_API_KEY"] = self.config["env"]["ncbi_api"].get("NCBI_API_KEY", "")

        print("Configuration loaded successfully.")
        # print(os.getenv("NCBI_API_KEY"))

        return self.config
        # # Optional: Print to verify environment variables
        # print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))
        # # print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY"))
        # print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT")) 
        # print("OLLAMA_HOST:", os.getenv("OLLAMA_HOST"))
        # print("OLLAMA_NUM_PARALLEL:", os.getenv("OLLAMA_NUM_PARALLEL"))
        # # print("AZURE_OPENAI_API_KEY:", os.getenv("AZURE_OPENAI_API_KEY"))
        # print("MISTRAL:", os.getenv("MISTRAL"))