import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class Retriever:
    """
    Handles indexing and searching the corpus using TF-IDF.
    """
    def __init__(self, documents: List[Dict[str, str]]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        if documents:
            self.corpus_contents = [doc['content'] for doc in documents]
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_contents)
        else:
            self.corpus_contents = []
            self.tfidf_matrix = None

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Searches the corpus for the query and returns top_k relevant documents.
        """
        if self.tfidf_matrix is None or not query:
            return []

        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get indices of top_k similarities
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score > 0:
                    results.append({
                        'content': self.documents[idx]['content'],
                        'path': self.documents[idx]['path'],
                        'score': float(score)
                    })
            
            return results
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

    def get_confidence_score(self, results: List[Dict]) -> float:
        """Returns the confidence score of the top result."""
        if not results:
            return 0.0
        return results[0]['score']
