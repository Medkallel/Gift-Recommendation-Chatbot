__import__("pysqlite3")
import sys
import time
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import pysqlite3
import os
import csv
import time
import streamlit as st
from langchain_chroma import Chroma
from langchain_together import TogetherEmbeddings
from langchain_community.document_loaders import CSVLoader

import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

st.set_page_config(page_icon="🎁", page_title="Gift Recommendation Assistant")

# Constants
TMP_FILE_PATH = "./tmp/data.csv"
EMBEDDINGS_MODEL_NAME = "togethercomputer/m2-bert-80M-2k-retrieval"
VECSTORE_PERSIST_DIRECTORY = "./chroma_vectorstore/"


def load_and_store_embedded_documents(
    file_path,
    embeddings_model_name=EMBEDDINGS_MODEL_NAME,
    vecstore_persist_directory=VECSTORE_PERSIST_DIRECTORY,
):
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
    embeddings = TogetherEmbeddings(
        model=embeddings_model_name, api_key=st.secrets["TOGETHER_API_KEY"]
    )
    # Initialize the progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Track the total number of documents
    total_docs = len(data)
    start_time = time.time()

    for idx, doc in enumerate(data):
        # Process each document one at a time
        vectorstore = Chroma.from_documents(
            documents=[doc],  # Pass a single document as a list
            embedding=embeddings,
            persist_directory=VECSTORE_PERSIST_DIRECTORY,
        )
        
        # Update the progress bar
        progress_bar.progress((idx + 1) / total_docs)
        
        # Calculate elapsed time and documents per second
        elapsed_time = time.time() - start_time
        docs_per_sec = (idx + 1) / elapsed_time
        remaining_docs = total_docs - (idx + 1)
        estimated_time_remaining = remaining_docs / docs_per_sec if docs_per_sec > 0 else float('inf')
        
        # Convert estimated time remaining to hh:mm:ss format
        hrs, rem = divmod(estimated_time_remaining, 3600)
        mins, secs = divmod(rem, 60)
        time_remaining_str = f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"
        
        # Update the status text
        status_text.text(
            f"Processing document {idx + 1}/{total_docs} "
            f"({docs_per_sec:.2f} docs/sec, "
            f"Estimated time remaining: {time_remaining_str})"
        )

    return vectorstore


st.title("Products Catalogue ⚙️")
st.info(
    "Upload a CSV file with the products catalogue to get started.  \nThe CSV file should contain the following columns in this order: {'product_id', [Product Feature & Description Columns...]}"
)

if "embed" not in st.session_state:
    st.session_state.embed = False

csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

if st.button("Load & Embed Documents", type="primary"):
    st.session_state.embed = True

if st.session_state.embed:
    st.error(
        "This operation will overwrite the existing embedded documents. Are you sure you want to proceed? RECOVERY IS NOT POSSIBLE"
    )
    if st.button("Yes, Proceed", type="secondary"):
        st.info(
            "This may take a while depending on the size of the CSV file. Please be patient."
        )
        with st.spinner("Embedding Documents..."):
            try:
                if not os.path.exists("./tmp"):
                    os.makedirs("./tmp")
                with open(TMP_FILE_PATH, "wb") as f:
                    f.write(csv_file.getvalue())
                load_and_store_embedded_documents(TMP_FILE_PATH)
                os.remove(TMP_FILE_PATH)
                st.success("Documents Embedded Successfully!")
                time.sleep(3)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        st.rerun()
