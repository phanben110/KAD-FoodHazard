from typing import List
from metapub import PubMedFetcher
from src.abstract_retrieval.AbstractRetriever import AbstractRetriever
from src.abstract_retrieval.AbstractRetriever import ScientificAbstract
from src.abstract_retrieval.utils import *
from src.utils import get_logger
# from config.logging_config import get_logger


class PubMedAbstractRetriever(AbstractRetriever):
    def __init__(self, pubmed_fetch_object: PubMedFetcher):
        self.pubmed_fetch_object = pubmed_fetch_object
        self.logger = get_logger(__name__)

    def _simplify_pubmed_query(self,llm, query: str, simplification_function: callable = simplify_pubmed_query) -> str:
        return simplification_function(llm, query)

    def _get_abstract_list(self, llm, query: str, simplify_query: bool = True) -> List[str]:
        """ Fetch a list of PubMed IDs for the given query. """
        if simplify_query:
            #self.logger.info(f'Trying to simplify scientist query {query}')
            #print(f'Trying to simplify scientist query {query}')
            #print(f'Trying to simplify scientist query: {query}')
            query_simplified = self._simplify_pubmed_query(llm, query)

            if query_simplified != query:
                #self.logger.info(f'Initial query simplified to: {query_simplified}')
                #print(f'Initial query simplified to: {query_simplified}')
                query = query_simplified
            else:
                #print(f'Initial query is simple enough and does not need simplification.')
                self.logger.info('Initial query is simple enough and does not need simplification.')

        #self.logger.info(f'Searching abstracts for query: {query}') 
        #print(f'Searching abstracts for query: {query}')
        return self.pubmed_fetch_object.pmids_for_query(query, retmax=2), query 

    def _get_abstracts(self, pubmed_ids: List[str]) -> List[ScientificAbstract]:
        """ Fetch PubMed abstracts  """
        #print(f'Fetching abstract data for following pubmed_ids: {pubmed_ids}')
        #self.logger.info(f'Fetching abstract data for following pubmed_ids: {pubmed_ids}')
        scientific_abstracts = []
        
        for id in pubmed_ids:
            abstract = self.pubmed_fetch_object.article_by_pmid(id)
            if abstract.abstract is None:
                continue

            #print(f' Doi {abstract.doi} \n Title {abstract.title} \n Authors {abstract.authors} \n Year {abstract.year} \n Abstract {abstract.abstract}')
            
            abstract_formatted = ScientificAbstract(
                doi=abstract.doi,
                title=abstract.title,
                authors=', '.join(abstract.authors),
                # authors= abstract.authors,
                year=abstract.year,
                abstract_content=abstract.abstract
            )
            scientific_abstracts.append(abstract_formatted)

        #print(f'Total of {len(scientific_abstracts)} abstracts retrieved.')
        #self.logger.info(f'Total of {len(scientific_abstracts)} abstracts retrieved.')
        
        return scientific_abstracts

    def get_abstract_data(self, llm, scientist_question: str, simplify_query: bool = True) -> List[ScientificAbstract]:
        """  Retrieve abstract list for scientist query. """
        #print(f'Scientist: {scientist_question}') 
        #print(f'Simplify query: {simplify_query}')
        pmids, query = self._get_abstract_list(llm, scientist_question, simplify_query)
        abstracts = self._get_abstracts(pmids)
        return abstracts, query