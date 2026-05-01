import os
import glob
from typing import List, Dict

class CorpusLoader:
    """
    Loads markdown documentation from the specified directory recursively.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.documents = []

    def load_corpus(self) -> List[Dict[str, str]]:
        """
        Recursively loads all markdown files from the data directory.
        Returns a list of dictionaries with 'path' and 'content'.
        """
        self.documents = []
        # Use glob to find all .md files recursively
        search_pattern = os.path.join(self.data_dir, "**", "*.md")
        md_files = glob.glob(search_pattern, recursive=True)
        
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        self.documents.append({
                            'path': file_path,
                            'content': content,
                            'filename': os.path.basename(file_path)
                        })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return self.documents

if __name__ == "__main__":
    # Test loader
    loader = CorpusLoader("data")
    docs = loader.load_corpus()
    print(f"Loaded {len(docs)} documents.")
