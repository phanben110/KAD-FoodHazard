# template_prompt.py

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

class PromptTemplates:
    def __init__(self):

        self._sem_eval_2025_task9_prompt_template_rag = """
You are tasked with paraphrasing the given passage for data augmentation. Follow these rules:

1. Retain exact terms, names, numbers, and specific technical phrases related to food hazards and food products.
2. Contextual Accuracy:
   - Ensure the paraphrase is accurate and aligns with the context provided. Do not alter the meaning or factual content.

<context>
{context}
</context>

Here is your task:  
Given the input:  
"{passage}"  

Paraphrase it according to the rules above, ensuring the augmented text is consistent with the context.  
Please follow the rules above and paraphrase the given passage. Output only the paraphrased result, with no additional comments."""

        self._sem_eval_2025_task9_prompt_template = (
            "system",
            "You are tasked with paraphrasing the given passage for data augmentation. Follow these rules:\n"
            "1. Retain exact terms, names, numbers, and specific technical phrases related to food hazards and food products. \n\n"
            "Task:\n"
            'Paraphrase the following passage while adhering to the above rules. Output only the paraphrased result. \n\n'
            '"{passage}"'
        )

        self._evaluation_prompt_template = '''
You are evaluating data augmentation through paraphrasing a given text. The evaluation must follow the scoring system below:

0: The data augmentation result is exactly the same as the reference text. 
1: The data augmentation result is completely unrelated to the reference text.
2: The data augmentation result has minor relevance but does not align with the reference text.
3: The data augmentation result has moderate relevance but contains inaccuracies.
4: The data augmentation result aligns with the reference text but has minor errors or omissions.
5: The data augmentation result is completely accurate and aligns perfectly with the reference text.

Here is your task:

Given the Original text: 
{original_text}

Given the Augmented text: 
{augmented_text}

Please evaluate the paraphrased text according to the scoring system above. Output only the score from 0 to 5, no other comments needed.''' 


    def get_sem_eval_2025_task9_prompt_template(self):
        return self._sem_eval_2025_task9_prompt_template

    def get_sem_eval_2025_task9_prompt_template_rag(self):
        return self._sem_eval_2025_task9_prompt_template_rag  

    def get_evaluation_prompt_template(self):
        return self._evaluation_prompt_template 

    def create_chat_prompt_template(self, prompt):
        return ChatPromptTemplate.from_messages([prompt, MessagesPlaceholder(variable_name="messages")])
