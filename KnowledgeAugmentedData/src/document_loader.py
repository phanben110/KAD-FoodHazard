from typing import AsyncIterator, Iterator, List
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class DocumentLoader(BaseLoader):
    """An example document loader that reads a list of files line by line."""

    def __init__(self, file_paths: List[str]) -> None:
        """Initialize the loader with a list of file paths.

        Args:
            file_paths: A list of paths to the files to load.
        """
        self.file_paths = file_paths

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads each file in the list line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        for file_path in self.file_paths:
            with open(file_path, 'r') as file:
                current_id = None
                current_text = ""

                for line in file:
                    # Check if the line contains a title (|t|)
                    if '|t|' in line:
                        current_id = line.split('|')[0]  # Extract the ID before |t|
                        current_text = line.split('|t|')[1].strip()  # Extract text after |t|

                    # Check if the line contains an abstract (|a|)
                    elif '|a|' in line and current_id:
                        abstract = line.split('|a|')[1].strip()  # Extract text after |a|
                        combined_text = current_text + " " + abstract  # Concatenate title and abstract

                        yield Document(
                            page_content=combined_text,
                            metadata={"PMID": current_id, "source": file_path},
                        )
                        current_id = None  # Reset for the next document

    async def alazy_load(
        self,
    ) -> AsyncIterator[Document]:  # <-- Does not take any arguments
        """An async lazy loader that reads each file in the list line by line."""
        # Requires aiofiles
        # Install with `pip install aiofiles`
        # https://github.com/Tinche/aiofiles
        import aiofiles

        for file_path in self.file_paths:
            async with aiofiles.open(file_path, 'r') as file:
                current_id = None
                current_text = ""

                async for line in file:
                    # Check if the line contains a title (|t|)
                    if '|t|' in line:
                        current_id = line.split('|')[0]  # Extract the ID before |t|
                        current_text = line.split('|t|')[1].strip()  # Extract text after |t|

                    # Check if the line contains an abstract (|a|)
                    elif '|a|' in line and current_id:
                        abstract = line.split('|a|')[1].strip()  # Extract text after |a|
                        combined_text = current_text + " " + abstract  # Concatenate title and abstract

                        yield Document(
                            page_content=combined_text,
                            metadata={"PMID": current_id, "source": file_path},
                        )
                        current_id = None  # Reset for the next document


class DocumentLoader_2(BaseLoader):
    """An example document loader that reads a list of files line by line."""

    def __init__(self, file_paths: List[str]) -> None:
        """Initialize the loader with a list of file paths.

        Args:
            file_paths: A list of paths to the files to load.
        """
        self.file_paths = file_paths

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads each file in the list line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        for file_path in self.file_paths:
            with open(file_path, 'r') as file:
                current_id = None
                current_text = ""

                for line in file:
                    # Check if the line contains a title (|t|)
                    if '|t|' in line:
                        current_id = line.split('|')[0]  # Extract the ID before |t|
                        current_text = line.split('|t|')[1].strip()  # Extract text after |t|

                    # Check if the line contains an abstract (|a|)
                    elif '|a|' in line and current_id:
                        abstract = line.split('|a|')[1].strip()  # Extract text after |a|
                        combined_text = current_text + " " + abstract  # Concatenate title and abstract

                        yield Document(
                            page_content=combined_text,
                            metadata={
                                "source": file_path, 
                                "title": current_id, 
                                "authors": "Opensource dataset", 
                                "year_of_publication" : 2024
                                },
                                )
                        current_id = None  # Reset for the next document

    async def alazy_load(
        self,
    ) -> AsyncIterator[Document]:  # <-- Does not take any arguments
        """An async lazy loader that reads each file in the list line by line."""
        # Requires aiofiles
        # Install with `pip install aiofiles`
        # https://github.com/Tinche/aiofiles
        import aiofiles

        for file_path in self.file_paths:
            async with aiofiles.open(file_path, 'r') as file:
                current_id = None
                current_text = ""

                async for line in file:
                    # Check if the line contains a title (|t|)
                    if '|t|' in line:
                        current_id = line.split('|')[0]  # Extract the ID before |t|
                        current_text = line.split('|t|')[1].strip()  # Extract text after |t|

                    # Check if the line contains an abstract (|a|)
                    elif '|a|' in line and current_id:
                        abstract = line.split('|a|')[1].strip()  # Extract text after |a|
                        combined_text = current_text + " " + abstract  # Concatenate title and abstract

                        yield Document(
                            page_content=combined_text,
                            metadata={"source": file_path, 
                                      "title": current_id, 
                                      "authors": "Opensource dataset", 
                                      "year_of_publication" : 2024
                                      },
                        )
                        current_id = None  # Reset for the next document