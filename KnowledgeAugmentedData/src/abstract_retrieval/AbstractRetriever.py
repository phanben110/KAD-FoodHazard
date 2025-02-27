from abc import ABC, abstractmethod
from typing import List

from typing import Optional
from pydantic import BaseModel


class ScientificAbstract(BaseModel):
    doi: Optional[str]
    title: Optional[str]
    authors: Optional[list]
    year: Optional[int]
    abstract_content: str

class AbstractRetriever(ABC):

    @abstractmethod
    def get_abstract_data(self, scientist_question: str) -> List[ScientificAbstract]:
        """ Retrieve a list of scientific abstracts based on a given query. """
        raise NotImplementedError
    


class ScientificAbstract(BaseModel):
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    abstract_content: str

class UserQueryRecord(BaseModel):
    user_query_id: str
    user_query: str