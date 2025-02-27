import pandas as pd
import json
import re
import warnings
warnings.filterwarnings("ignore")
from src.utils import *  

class MergerDataSemEval_2025_task9:
    def __init__(self, csv_file, json_files, matching_threshold=0.5, evaluation_threshold=3, auto_correct=False):
        self.csv_file = csv_file
        self.json_files = json_files  # Expecting a list of JSON files
        self.auto_correct = auto_correct
        self.cleaned_df = None
        self.raw_df = None 
        self.augmentation_df = None 
        self.number_of_augmentations = None
        self.full_df = None
        # self.column_names = ["PMID", "Type1", "Type2", "Identifier1", "Identifier2", "BoolCls", "LabelCls", "Document", "Relation", "Novelty"]
        self.column_names_aug = ["SampleID", "year", "month", "day", "country","title", "text", "hazard-category", "product-category", "hazard", "product", "Document" ]
        self.matching_threshold = matching_threshold
        self.evaluation_threshold = evaluation_threshold

    def load_csv(self):
        # Load and clean the TSV data
        self.raw_df = pd.read_csv(self.csv_file)
        self.raw_df = self.raw_df.reset_index().rename(columns={'index': 'SampleID'})
        self.raw_df['SampleID'] = self.raw_df['SampleID'].astype(str) 
        self.raw_df["Document"] = self.raw_df['title'] + " /n " + self.raw_df['text'] 
        # self.cleaned_df = self.raw_df[self.raw_df['pair type'] != 'false']  
        self.cleaned_df = self.raw_df 
        # self.raw_df['SampleID'] = self.raw_df['PMID'].astype(str) + "_" + self.raw_df.index.astype(str)
        return self.cleaned_df

    def load_json(self):
        # Initialize an empty list to store DataFrames for each JSON file
        all_aug_dfs = []
        
        # Loop over the list of JSON files and load them one by one
        for json_file in self.json_files:
            
            with open(json_file, 'r') as file:
                augmentation_data = json.load(file)
            
            # Convert the JSON to a DataFrame
            aug_df = pd.DataFrame.from_dict(augmentation_data, orient='index').reset_index()
            aug_df.rename(columns={'index': 'SampleID', 'original_text': 'Document', 'augmented_text': 'Augmentation', 
                                   'matching_score': 'MatchingScore', 'evaluation_score': 'EvaluationScore'}, inplace=True)
            # Append the DataFrame to the list
            all_aug_dfs.append(aug_df)
        
        # Concatenate all the DataFrames from the list into one DataFrame
        full_aug_df = pd.concat(all_aug_dfs, ignore_index=True) 
        self.number_of_augmentations = len(full_aug_df) 
        # print("Number of augmentation: ", len(full_aug_df)) 
        full_aug_df["Augmentation"] = full_aug_df["Augmentation"].apply(self.extract_text_after_delimiter) 
        
        return full_aug_df

    def merge_data(self, df_cleaned, aug_df):
        # Merge the cleaned TSV data with the augmentation data 
        merged_df = pd.merge(df_cleaned, aug_df, on=['SampleID'], how='outer')
        
        # Fill NaN values in 'EvaluationScore' and apply the score extraction function
        merged_df['EvaluationScore'] = merged_df['EvaluationScore'].fillna(0)
        merged_df['EvaluationScore'] = merged_df['EvaluationScore'].apply(self.extract_score)
        
        # Drop 'Document' and rename 'Augmentation' to 'Document'
        # merged_df = merged_df.drop(columns=['Document'])
        merged_df = merged_df.rename(columns={'Augmentation': 'Document'})
        return merged_df

    def merge_data_v2(self, df_cleaned, aug_df):
        # Merge the cleaned TSV data with the augmentation data 
        merged_df = pd.merge(df_cleaned, aug_df, on=['SampleID'], how='outer')
        
        
        # Drop 'Document' and rename 'Augmentation' to 'Document'
        # merged_df = merged_df.drop(columns=['Document'])
        merged_df = merged_df.rename(columns={'Augmentation': 'Document'})
        return merged_df

    @staticmethod
    def extract_score(value):
        # Extract the first number between 0 and 5 from the string
        match = re.search(r'[0-5]', str(value))
        if match:
            return int(match.group(0))
        return 0
    
    @staticmethod
    def extract_text_after_delimiter(text):
        # Remove triple quotes if present
        text = text.replace('"""', '').replace("'''", '')
        
        # Find the position of the first occurrence of '\n\n' or '\n'
        delimiter_pos = text.find('\n\n')
        
        if delimiter_pos == -1:
            delimiter_pos = text.find('\n')
        
        if delimiter_pos != -1:
            # Extract the text after the delimiter
            extracted_text = text[delimiter_pos + 2:].strip() if text[delimiter_pos:delimiter_pos + 2] == '\n\n' else text[delimiter_pos + 1:].strip()
        else:
            # If no delimiter is found, use the original text
            extracted_text = text
        
        # Remove remaining newline characters to ensure text is a single line
        extracted_text = extracted_text.replace('\n', ' ').replace('\r', '')

        return extracted_text


    def filter_data(self, df):
        # Apply threshold filters on MatchingScore and EvaluationScore
        #filtered_df = df[(df['MatchingScore'] >= self.matching_threshold) & (df['EvaluationScore'] >= self.evaluation_threshold)]
        filtered_df = df[(df['MatchingScore'] >= self.matching_threshold) & (df['EvaluationScore'] >= self.evaluation_threshold)]
        return filtered_df

    def concatenate_data(self, augmentation_df):
        self.raw_df['MatchingScore'] = 100
        self.raw_df['EvaluationScore'] = 100
        # Concatenate the original and augmented DataFrames
        concatenated_df = pd.concat([augmentation_df, self.raw_df], axis=0, ignore_index=True)
        return concatenated_df 
    
    def analyze_success_rate(self):
        # Get the total number of augmentations before filtering
        # total_augmentations = len(self.augmentation_df)

        # Get the filtered augmentations after applying thresholds
        successful_augmentations = self.augmentation_df[
            (self.augmentation_df['MatchingScore'] >= self.matching_threshold) &
            (self.augmentation_df['EvaluationScore'] >= self.evaluation_threshold)
        ]

        # Calculate the number of successful augmentations
        successful_count = len(successful_augmentations)

        # Calculate the success rate
        success_rate = (successful_count / self.number_of_augmentations) * 100 if self.number_of_augmentations > 0 else 0

        # Print or return the results
        print(f"Total augmentations: {self.number_of_augmentations}")
        print(f"Successful augmentations: {successful_count}")
        print(f"Success rate: {success_rate:.2f}%")
        
        return success_rate, successful_count, self.number_of_augmentations
    
    # def save_to_csv(self, output_file):
    #     # Define the export order of the columns
    #     export_columns = ["SampleID", "entity e1", "entity e2", "pair type", "Document"]
    #     self.full_df = self.full_df.sort_values(by='SampleID') 
    #     # Ensure the correct column order and export without a header
    #     self.full_df[export_columns].to_csv(output_file,index=False)
    #     print(f"File saved as {output_file}")

    def save_to_csv(self, output_file):
        # Convert 'SampleID' to int for correct sorting
        self.full_df['SampleID'] = self.full_df['SampleID'].astype(int)
        self.full_df['Full Sentence'] = self.full_df['Document']

        # Define the export order of the columns
        export_columns = self.column_names_aug
        
        # Sort by 'SampleID'
        self.full_df = self.full_df.sort_values(by='SampleID')
        
        # Ensure the correct column order and export to CSV without a header
        self.full_df[export_columns].to_csv(output_file, index=False)
        
        # Replace 'Document' column with 'Full Sentence' after exporting


        print(f"File saved as {output_file}")

    def replace_placeholders_in_df(self, row):
        document = row['Document']
        entity1 = row['entity e1']
        entity2 = row['entity e2']
        
        # Replace placeholders
        document = replace_placeholder(document, f'@{entity1}$', entity1)
        document = replace_placeholder(document, f'@{entity2}$', entity2)
        
        return document

    def process(self):
        # Main method to run the entire process
        df_cleaned = self.load_csv()
        aug_df = self.load_json()
        self.augmentation_df = self.merge_data(df_cleaned, aug_df) 

        self.analyze_success_rate()
        
        # # Apply filtering based on the thresholds
        self.augmentation_df = self.filter_data(self.augmentation_df)
        
        # Concatenate original and augmented data
        self.full_df = self.concatenate_data(self.augmentation_df) 
        # self.full_df["Document"] = self.full_df ["Document"].apply(self.extract_text_after_delimiter)
        # self.full_df["Document"] = self.full_df.apply(self.replace_placeholders_in_df, axis=1)
                
        return self.full_df 
    
    def process_eval_v2(self):
        # Main method to run the entire process
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        df_cleaned = self.load_csv()
        aug_df = self.load_json()

        aug_df["Augmentation"] = aug_df["Augmentation"].apply(format_input_text)
        aug_df["Document"] = aug_df["Document"].apply(format_input_text) 
        
        aug_df["MatchingScore"] = aug_df.apply(lambda row: check_word_count_difference(row['Augmentation'], row['Document']), axis=1)
        aug_df["EvaluationScore"] = aug_df.apply(lambda row: calculate_semantic_similarity(row['Augmentation'], row['Document'], model), axis=1)
        print(aug_df["EvaluationScore"])

        self.augmentation_df = self.merge_data_v2(df_cleaned, aug_df) 

        self.analyze_success_rate()
        
        # # Apply filtering based on the thresholds
        self.augmentation_df = self.filter_data(self.augmentation_df)
        
        # Concatenate original and augmented data
        self.full_df = self.concatenate_data(self.augmentation_df) 
        # self.full_df["Document"] = self.full_df ["Document"].apply(self.extract_text_after_delimiter)
        # self.full_df["Document"] = self.full_df.apply(self.replace_placeholders_in_df, axis=1)
                
        return self.full_df  
