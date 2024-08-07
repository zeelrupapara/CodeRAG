import ollama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tiktoken


from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)


def get_language_from_extension(file_path):
    """
    Detect the programming language from a file based on the file extension.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The detected programming language name, or 'Unknown' if the extension is not recognized.
    """
    # Define a dictionary mapping file extensions to programming language names
    extension_to_language = {
    '.py': 'python',
    '.js': 'js',
    '.java': 'java',
    '.cpp': 'cpp',
    '.c': 'c',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.go': 'go',
    '.rs': 'rust',
    '.scala': 'scala',
    '.kt': 'kotlin',
    '.hs': 'haskell',
    '.ts': 'ts',
    '.sh': 'bash',
    '.sql': 'sql',
    '.html': 'html',
    '.css': 'css',
    '.md': 'markdown',
    '.json': 'json',
    '.xml': 'xml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.ini': 'ini',
    '.csv': 'csv',
    '.tex': 'latex',
    '.r': 'r',
    '.m': 'matlab',
    '.ipynb': 'jupyter notebook',
    }

    # Get the file extension
    file_extension = f".{file_path.split('.')[-1].lower()}"

    # Check if the extension is in the dictionary
    if file_extension in extension_to_language:
        return extension_to_language[file_extension]
    else:
        return 'Unknown'
    

def list_files_in_directory(directory):
    # List to store the full file paths
    file_paths = []

    # Walk through the directory
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Create the full file path
            file_path = os.path.join(root, filename)
            # Add it to the list
            file_paths.append(file_path)

    return file_paths


def code_splitter(file_path, language):
  return RecursiveCharacterTextSplitter.from_language(
      language=language,
      chunk_size=4000,
      chunk_overlap=0,
  )


def split_and_chunk(file_path, language):
  splitter = code_splitter(file_path, language)
  with open(file_path, 'r') as f:
      code_snippets = f.read()
  docs = splitter.split_text(code_snippets)
  return docs


def get_embeddings(sentences):
  model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
  embeddings = model.encode(sentences)
  return embeddings


def count_tokens(text: str, model: str = "gpt-4"):
    # Load the tokenizer for the specified model
    encoding = tiktoken.encoding_for_model(model)
    
    # Tokenize the input text
    tokens = encoding.encode(text)
    
    # Return the number of tokens
    return len(tokens)


def get_token_count(text, model_name="gpt-4", ):
    num_tokens = count_tokens(text, model_name)
    return num_tokens


chunks_directory = []
directory_path = "input"
file_paths = list_files_in_directory(directory_path)

for file_path in file_paths:
  try:
    language = get_language_from_extension(file_path)
    docs = split_and_chunk(file_path, language)
    for doc in docs:
      embeddings = get_embeddings(doc)
      chunks_directory.append({
          'language': language,
          'tokens': get_token_count(doc),
          'file_path': file_path,
          'doc': doc,
          'embeddings': embeddings
      })
    print(f"Processed {file_path}")
  except Exception as e:
    print(f"Not Suppoerted Language: {language} in {file_path}")
    continue                
  

df = pd.DataFrame(chunks_directory)
df

# threshold = 500  # Example threshold value

# # Group by language and sum tokens for each language
# language_tokens = df.groupby('language')['tokens'].sum().reset_index()

# # Create subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# # First bar graph: Token Distribution Across Documents
# ax1.bar(df['doc'].apply(lambda x: x[:10]), df['tokens'])
# ax1.set_xlabel('Document')
# ax1.set_ylabel('Number of Tokens')
# ax1.set_title('Token Distribution Across Documents')
# ax1.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
# ax1.legend()
# ax1.set_xticklabels(df['doc'].apply(lambda x: x[:10]), rotation=45, ha='right')

# # Second bar graph: Token Distribution Across Languages
# ax2.bar(language_tokens['language'], language_tokens['tokens'])
# ax2.set_xlabel('Language')
# ax2.set_ylabel('Number of Tokens')
# ax2.set_title('Token Distribution Across Languages')
# ax2.set_xticklabels(language_tokens['language'], rotation=45, ha='right')

# # Adjust layout
# plt.tight_layout()

# # Show the plots
# plt.show()


def rank_chunks_by_similarity(user_query, chunks):
    # Embed the user query
    query_embedding = get_embeddings([user_query])[0]

    # Extract embeddings from the chunks dictionary
    chunk_embeddings = [chunk['embeddings'] for chunk in chunks]

    # Calculate cosine similarity between the user query and chunk embeddings
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]

    # Create a list of chunks with their similarity scores
    chunks_with_scores = [{'chunk': chunk, 'similarity_score': score} for chunk, score in zip(chunks, similarities)]

    # Sort the list of chunks by similarity score in descending order
    sorted_chunks_with_scores = sorted(chunks_with_scores, key=lambda x: x['similarity_score'], reverse=True)

    return sorted_chunks_with_scores


def serach_and_response():
    user_query = "Write a optimize solution for utc convert function in utc file"
    ranked_chunks = rank_chunks_by_similarity(user_query, chunks_directory)
    ranked_chunks_directory = []
    # Print sorted chunks and their similarity scores
    for ranked_chunk in ranked_chunks:
        ranked_chunks_directory.append({
            'language': ranked_chunk['chunk']['language'],
            'tokens': ranked_chunk['chunk']['tokens'],
            'file_path': ranked_chunk['chunk']['file_path'],
            'doc': ranked_chunk['chunk']['doc'],
            'embeddings': ranked_chunk['chunk']['embeddings'],
            'similarity_score': ranked_chunk['similarity_score']
        })
    ranked_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
    

    # models = ollama.list()
    # print(models)

    # need to get top 5 response from the ranks_chunks and create messages for ollama
    messages = []

    for ranked_chunk in ranked_chunks_directory[:1]:
      print(ranked_chunk['doc'])
      messages.append({'role': 'user', 'content': f'file name:: {ranked_chunk["file_path"]} and used programmiong language:{ranked_chunk["language"]}, Below is a content of code: {ranked_chunk["doc"]}, Todo when i told that start response then start until say remember.'})
    messages.append({'role': 'user', 'content': f"TODO: {user_query} and start response"})
    # User Query 
    # messages.append({'role': 'user', 'content': f'{user_query}'})

    # ollama chat
    stream = ollama.chat(
        model='llama2',
        messages=messages,
        stream=True,
    )

    print("------------------------output-------------------------")
    for chunk in stream:
      print(chunk['message']['content'], end='', flush=True)

    # client = OpenAI()

    # stream = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=messages,
    #     stream=True,
    # )
    # for chunk in stream:
    #     if chunk.choices[0].delta.content is not None:
    #         print(chunk.choices[0].delta.content, end="")

    # return serach_and_response()

serach_and_response()
