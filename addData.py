import csv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document Loading
file_path = "./Data/beauty_test_amazon.csv"
csv.field_size_limit(10**6)
loader = CSVLoader(
    file_path=file_path,
    csv_args={
        "delimiter": ",",
    },
)
data = loader.load()

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=100, add_start_index=True
)
all_splits = text_splitter.split_documents(data)

# Embeddings
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Vector Store
persist_directory = './chroma_vectorstore/'
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=persist_directory)
