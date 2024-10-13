__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import pysqlite3
import os
import csv
import time
import streamlit as st
from langchain_chroma import Chroma
from langchain_together import TogetherEmbeddings
from langchain_community.document_loaders import CSVLoader


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
        model=EMBEDDINGS_MODEL_NAME, api_key=st.secrets["TOGETHER_API_KEY"]
    )
    # Vectorstore
    vectorstore = Chroma.from_documents(
        documents=data,
        embedding=embeddings,
        persist_directory=vecstore_persist_directory,
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
