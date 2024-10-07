import os
import pickle
import uuid
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi  # Import BM25

import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
#from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain


class SemanticSearchEngine:
    def __init__(self, model_name="msmarco-distilbert-base-v4", similarity_threshold=0.7, batch_size=200):
        """Initialize the semantic search engine with the given model and settings."""
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.data_frame = pd.DataFrame()
        self.llm = Ollama(model="llama3", base_url="http://127.0.0.1:11434")
        self.embed_model = OllamaEmbeddings(
           model="llama3",
            base_url='http://127.0.0.1:11434'
        )

    def encode_in_batches(self, paragraphs):
        """Encode the input text in batches to optimize memory usage."""
        paragraph_embeddings = []
        total_paragraphs = len(paragraphs)

        if total_paragraphs < self.batch_size:
            #return self.embed_model.encode(paragraphs, convert_to_numpy=True)
            return self.embed_model.embed_documents(paragraphs, convert_to_numpy=True)

        for i in range(0, total_paragraphs, self.batch_size):
            batch_text = paragraphs[i: i + self.batch_size]
            #batch_embeddings = self.embed_model.encode(batch_text, convert_to_numpy=True)
            batch_embeddings = self.embed_model.embed_documents(batch_text)
            paragraph_embeddings.append(batch_embeddings)

        return np.vstack(paragraph_embeddings)

    def split_text_by_similarity(self, text):
        """Split text into chunks by calculating semantic similarity between paragraphs."""
        paragraphs = text.split('\n')
        paragraph_embeddings = self.encode_in_batches(paragraphs)
        chunks = []
        similarity_matrix = cosine_similarity(paragraph_embeddings)

        for idx, paragraph in enumerate(paragraphs):
            if idx == 0:
                chunks.append([paragraph])
            else:
                sim_score = similarity_matrix[idx - 1][idx]
                if sim_score > self.similarity_threshold:
                    chunks[-1].append(paragraph)
                else:
                    chunks.append([paragraph])

        return [' '.join(chunk) for chunk in chunks]

    def load_and_process_corpus(self, corpus_directory, cache_filename='corpus_cache.pkl'):
        """Load corpus data, preprocess it, and cache the results for faster future access."""
        cache_path = Path(cache_filename)
        if cache_path.exists():
            with open(cache_path, 'rb') as cache_file:
                self.data_frame = pickle.load(cache_file)
            st.write("Loaded corpus data from cache.")
        else:
            content_list = []
            file_list = []
            chunk_ids = []
            embeddings_list = []
            document_texts = []
            movie_titles = []

            for file in os.listdir(corpus_directory):
                file_path = os.path.join(corpus_directory, file)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        text_content = f.read()
                        chunks = self.split_text_by_similarity(text_content)
                        #chunk_embeddings = self.embed_model.encode(chunks, convert_to_tensor=True)
                        chunk_embeddings = self.embed_model.embed_documents(chunks)
                        #chunk_embeddings = chunk_embeddings.cpu().detach().numpy()
                        chunk_embeddings = np.array(chunk_embeddings)  # Ensure this is a numpy array

                        for i, chunk in enumerate(chunks):
                            content_list.append(chunk)
                            file_list.append(file)
                            chunk_ids.append(str(uuid.uuid4().fields[-1])[:5])
                            embeddings_list.append(chunk_embeddings[i])
                            document_texts.append(text_content)
                            movie_titles.append(os.path.splitext(file)[0])

            if not (len(content_list) == len(file_list) == len(chunk_ids) == len(embeddings_list) == len(document_texts) == len(movie_titles)):
                raise ValueError("Data inconsistency detected. Ensure all lists have equal lengths.")

            data_dict = {
                "Chunk ID": chunk_ids,
                "Filename": file_list,
                "Text Chunks": content_list,
                "Embeddings": embeddings_list,
                "Full Document": document_texts,
                "Movie Title": movie_titles
            }
            self.data_frame = pd.DataFrame(data_dict)

            with open(cache_path, 'wb') as cache_file:
                pickle.dump(self.data_frame, cache_file)
            st.write("Corpus processed and saved to cache.")

        return self.data_frame

    def search(self, query):
        """Execute a semantic search using the query text and return matching results."""
        query_embedding = self.embed_model.embed_query(query)
        query_embedding = torch.tensor(query_embedding, dtype=torch.float32)  # Ensure query is float32

        # Convert corpus embeddings to float32
        corpus_embeddings = [torch.tensor(e, dtype=torch.float32) for e in self.data_frame["Embeddings"].tolist()]


        search_results = util.semantic_search(query_embedding, 
                                                np.array([np.array(e) for e in self.data_frame["Embeddings"].tolist()]),  # Convert to numpy arrays
                                                top_k=5)

        #chain = create_retrieval_chain(combine_docs_chain=llm,retriever=self.retriever)
    


        return search_results

    def search_bm25(self, query):
        """Execute a BM25 search and return matching results."""
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:5]  # Get top 5 results
        search_results = [{"corpus_id": idx, "score": scores[idx]} for idx in top_n]
        return [search_results]

    def format_search_results(self, search_results):
        """Format and return the search results for display."""
        result_chunks = []
        chunk_ids = []
        scores = []
        ranks = []
        document_texts = []
        movie_titles = []

        for rank, result in enumerate(search_results[0]):
            corpus_id = result['corpus_id']
            chunk_ids.append(self.data_frame["Chunk ID"].iloc[corpus_id])
            scores.append(result["score"])
            result_chunks.append(self.data_frame["Text Chunks"].iloc[corpus_id])
            document_texts.append(self.data_frame["Full Document"].iloc[corpus_id])
            movie_titles.append(self.data_frame["Movie Title"].iloc[corpus_id])
            ranks.append(rank + 1)

        results_df = pd.DataFrame({
            "Rank": ranks,
            "Score": scores,
            "Chunk ID": chunk_ids,
            "Text Chunk": result_chunks,
            "Full Document": document_texts,
            "Movie Title": movie_titles
        })

        combined_text = " ".join(result_chunks)
        return results_df, combined_text


# Streamlit app initialization and setup
def initialize_session_state():
    """Initialize session state variables in Streamlit."""
    if 'search_query' not in st.session_state:
        st.session_state.search_query = None
    if 'corpus_data' not in st.session_state:
        st.session_state.corpus_data = None


def main_app():
    """Main function to run the Streamlit interface."""
    st.title("Semantic Search Engine Application")

    initialize_session_state()
    semantic_search = SemanticSearchEngine()

    if st.session_state.search_query is None:
        query = st.text_input('Enter search query:')
        st.session_state.search_query = query
    else:
        query = st.text_input('Enter search query:', st.session_state.search_query)

    search_trigger = st.button('Search')
    corpus_path = r'/home/manju/semantic/corpus2'
    base_directory = r'/home/manju/semantic'
    cache_path = Path(base_directory) / "corpus_cache.pkl"

    if st.session_state.corpus_data is None:
        corpus_df = semantic_search.load_and_process_corpus(corpus_path, cache_filename=cache_path)
        corpus_df.to_csv(Path(base_directory) / "processed_corpus.csv")
    else:
        corpus_df = pd.read_csv(Path(base_directory) / "processed_corpus.csv")

    if search_trigger:
        search_results = semantic_search.search(query)
        results_df, combined_text = semantic_search.format_search_results(search_results)
        st.dataframe(results_df, use_container_width=True)



if __name__ == '__main__':
    main_app()
