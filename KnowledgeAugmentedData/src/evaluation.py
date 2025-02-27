# evaluation.py

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

class Evaluation:
    def __init__(self, llm, evaluationPromptTemplate ):
        self.llm = llm
        self.evaluation_prompt = ChatPromptTemplate.from_template(evaluationPromptTemplate) 
        self.chain_evaluation = LLMChain(llm=self.llm, prompt=self.evaluation_prompt)

    def evaluate_batch(self, inputs_val):
        return self.chain_evaluation.batch(inputs_val) 
    
    def evaluate_single(self, inputs_val):
        return self.chain_evaluation.invoke(inputs_val)
