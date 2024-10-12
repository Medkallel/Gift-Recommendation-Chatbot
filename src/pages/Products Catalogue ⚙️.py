import os
import csv
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader

st.set_page_config(page_icon="🎁", page_title="Gift Recommendation Assistant")

# Constants
EMBEDDINGS_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
VECSTORE_PERSIST_DIRECTORY = "./src/chroma_vectorstore"


def load_and_store_embedded_documents(file_path, embeddings_model_name=EMBEDDINGS_MODEL_NAME, vecstore_persist_directory=VECSTORE_PERSIST_DIRECTORY):
    # Document Loading
    csv.field_size_limit(10**6)
    loader = CSVLoader(
        file_path=file_path,
        csv_args={
            "delimiter": ",",
        },
        source_column="product_id",
    )
    data = loader.load()

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    # Vectorstore
    vectorstore = Chroma.from_documents(documents=data, embedding=embeddings, persist_directory=vecstore_persist_directory)
    
    return vectorstore


st.title("Products Catalogue ⚙️")
st.info("Upload a CSV file with the products catalogue to get started.  \nThe CSV file should contain the following columns in this order: {'product_id', [Product Feature & Description Columns...]}")
csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
if st.button("Load & Embed Documents",type="primary"):
    try:
        with st.spinner("Embedding Documents..."):
            os.remove(VECSTORE_PERSIST_DIRECTORY)
            load_and_store_embedded_documents(csv_file)
        st.success("Documents Embedded Successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")