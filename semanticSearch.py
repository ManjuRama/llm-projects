import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
import os
import uuid
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import util
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(page_title="Semantic Search Engine (Tom Hanks Movie Database)", page_icon="🐍", layout="wide")


## store session information
def set_session_state():
    """ """
    # default values
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'search' not in st.session_state:
        st.session_state.search = None
    if 'db' not in st.session_state:
        st.session_state.db = None


    # get parameters in url
    para = st.experimental_get_query_params()
    if 'search' in para:
        st.experimental_set_query_params()
        st.session_state.search = urllib.parse.unquote(para['search'][0])
    if 'db' in para:
        st.experimental_set_query_params()
        st.session_state.tags = para['tags'][0]
    if 'page' in para:
        st.experimental_set_query_params()

def chunk_text(text):
    #print (text[:20])
    # Split the input text into individual sentences.
    single_sentences_list = _split_sentences(text)
    # Combine adjacent sentences to form a context window around each sentence.
    combined_sentences = _combine_sentences(single_sentences_list)
    
    # Convert the combined sentences into vector representations using a neural network model.
    embeddings = convert_to_vector(combined_sentences)
    
    # Calculate the cosine distances between consecutive combined sentence embeddings to measure similarity.
    distances = _calculate_cosine_distances(embeddings)
    
    # Determine the threshold distance for identifying breakpoints based on the 80th percentile of all distances.
    breakpoint_percentile_threshold = 80
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    # Find all indices where the distance exceeds the calculated threshold, indicating a potential chunk breakpoint.
    indices_above_thresh = [i for i, distance in enumerate(distances) if distance > breakpoint_distance_threshold]
    # Initialize the list of chunks and a variable to track the start of the next chunk.
    chunks = []
    start_index = 0
    # Loop through the identified breakpoints and create chunks accordingly.
    for index in indices_above_thresh:
        chunk = ' '.join(single_sentences_list[start_index:index+1])
        chunks.append(chunk)
        start_index = index + 1
    
    # If there are any sentences left after the last breakpoint, add them as the final chunk.
    if start_index < len(single_sentences_list):
        chunk = ' '.join(single_sentences_list[start_index:])
        chunks.append(chunk)
    
    # Return the list of text chunks.
    return chunks


def _split_sentences(text):
    # Use regular expressions to split the text into sentences based on punctuation followed by whitespace.
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return sentences

def _combine_sentences(sentences):
    # Create a buffer by combining each sentence with its previous and next sentence to provide a wider context.
    combined_sentences = []
    for i in range(len(sentences)):
        combined_sentence = sentences[i]
        if i > 0:
            combined_sentence = sentences[i-1] + ' ' + combined_sentence
        if i < len(sentences) - 1:
            combined_sentence += ' ' + sentences[i+1]
        combined_sentences.append(combined_sentence)
    return combined_sentences

def convert_to_vector(texts):
    # Try to generate embeddings for a list of texts using a pre-trained model and handle any exceptions.
    try:
        embeddings = embedder.encode(texts, convert_to_tensor=True)
        return embeddings
    except Exception as e:
        print("An error occurred:", e)
        return np.array([])  # Return an empty array in case of an error

def _calculate_cosine_distances(embeddings):
    # Calculate the cosine distance (1 - cosine similarity) between consecutive embeddings.
    distances = []
    for i in range(len(embeddings) - 1):
        em1 =  embeddings[i].numpy()
        em2 = embeddings[i+1].numpy()

        similarity = cosine_similarity([em1], [em2])[0][0]
        distance = 1 - similarity
        distances.append(distance)
    return distances

CORPUS = r'C:\Users\MANJU\Documents\SV\corpus2'
DATA_DIR = r'C:\Users\MANJU\Documents\SV'
MODEL = 'msmarco-distilbert-base-v4'
embedder = SentenceTransformer(MODEL)


#user_input = st.text_input("Search Text", "Candy")

def search_text (query_text, embeddings):
  """
    key search function

  :param p1: describe about parameter p1
  :param p2: describe about parameter p2
  """ 
  #embeddings = get_corpus_embeddings (sentences)
  query = embedder.encode(query_text, convert_to_tensor=True)
  search_results = util.semantic_search(query, embeddings, top_k = 3)
  return search_results

def get_corpus (corpus_loc = CORPUS):
  """
   get corpus of all movie scripts
  :param p1: describe about parameter p1
  """ 

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20,
    separators=[
        "\n\n",
        "\n",
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    # Existing args
  )

  contents_array = []
  fileNames = []
  chunkIDs = []

  files = os.listdir(corpus_loc)
  for file in files:
    if os.path.isfile(os.path.join(CORPUS, file)):
        #print (file)
        f = open(os.path.join(CORPUS, file),'r')
        txt = f.read()
        print ('*' * 100, file)
        #texts = text_splitter.create_documents([txt])
        chunks = chunk_text (txt)

        for i, row in enumerate(chunks):
          contents_array.append(row)
          fileNames.append (file)
          chunkIDs.append (str(uuid.uuid4().fields[-1])[:5])
  f.close()


  embeddings = embedder.encode(contents_array, convert_to_tensor=True)
 
  print ("Len is: ", len(contents_array), " embeddings", len (embeddings))

  # create data results Dataframe
  data = {
    "Chunk ID": chunkIDs,
    "Filename": fileNames,
    "sentences": contents_array,
    "embeddings": list (embeddings),
   }
  df = pd.DataFrame (data)
  return df

def return_results (search_results, sentences):
  txt = []
  IDs = []
  scores = []

  df = pd.DataFrame(columns=["ID", "Score", "Result"])
  for index, result in enumerate(search_results[0]):
    print('*'*80)
    print(f'Search Rank: {index}, Relevance score: {result["score"]} ')
    print(sentences[result['corpus_id']])
    #str1 = str(index) + " : " + sentences[result['corpus_id']]
    IDs.append (str(index))
    scores.append (result["score"])
    txt.append (sentences[result['corpus_id']])
    
  str1 = " "
  df["ID"] = pd.Series(IDs)
  df["Score"] = pd.Series(scores)
  df["Result"] = pd.Series(txt)


  strFinal = str1.join(txt)
  return df, strFinal

def main():
# Page setup
    st.title("Semantic Search Engine")

    set_session_state()

    if st.session_state.search is None:
        search = st.text_input('Enter Search words:')
        st.session_state.search = search
    else:
        search = st.text_input('Enter Search words:', st.session_state.search)

    query_text = search
    goSearch = st.button('Search')
    sentences = []
    df = pd.DataFrame()

    if st.session_state.db is None:
        df = get_corpus()
        df.to_csv (DATA_DIR + r"\movies_db.csv")    
    else:
       df = pd.read_csv (DATA_DIR + r"\movies_db.csv")
    if goSearch:       
      sentences = df["sentences"]
      embeddings = df["embeddings"].tolist() 
      print (f"*********** query text is  {query_text}")
      search_results = search_text (query_text, embeddings)
      df, txt1 = return_results (search_results, sentences)
      #st.write('Results \n ', txt1)
      st.dataframe(df, use_container_width=True)
      print (df.head ())

if __name__ == '__main__':
    main()
