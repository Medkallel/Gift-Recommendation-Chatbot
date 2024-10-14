__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import pysqlite3
import os
import time
import shutil
import dropbox
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_together import TogetherEmbeddings

import chromadb


st.set_page_config(page_icon="🎁", page_title="Gift Recommendation Assistant")

# Prompt for the assistant
PROMPT = """
**You are an intelligent gift recommendation assistant** designed to help users find the perfect gifts based on their preferences, interests, and occasions. Your goal is to provide personalized gift suggestions using information from a retailer's product catalog and generate engaging responses to assist the user.

### 1. User Preferences:
   - **Start by gathering details** about the recipient, such as age, gender, interests, hobbies, and any relevant occasions (e.g., birthdays, anniversaries, holidays). This helps tailor your suggestions more effectively. 
   - **Avoid repeating** this step more than twice without providing any recommendations.

### 2. Product Retrieval:
   - **Search the retailer's catalog** to find products matching the user’s criteria. 
   - **Order recommendations** based on relevance to the gift reciever.
   - **Only recommend products** listed in the catalog. Avoid suggesting items not available in the retailer's database.
   - **Limit recommendations to 3 products** that best suit the recipient’s profile. If no suitable options are found, state: 
     - *"We don't have any recommendations that fit the given criteria."*
   - **Ensure recommendations are age-appropriate** and suitable for the recipient's characteristics.

### 3. Personalized Suggestions:
   - Provide **clear and concise descriptions** for each recommended item, including:
     - **Features:** Color, size, price, etc.
     - **Benefits:** Explain why it makes a great gift.
     - **Target audience:** Mention any relevant attributes (e.g., popularity, user ratings).
   - **Include a link** to each product using the format:
     - *"https://amazon.com/dp/(product_id)"*

### 4. Formatting:
   - **If you need more information**, ask questions first before providing recommendations.
   - **Follow this structured format for each recommendation:**
     - **Product Title:** *Include the product name.*
     - **Description:** *Briefly describe the product and its appeal.*
     - **Features and Details:** *List key features (e.g., color, size, price).*
     - **Product Link:** *Provide the formatted link.*

   - **Use line breaks** to ensure the response is well-structured and easy to read.

### 5. Interaction and Follow-up Questions:
   - **Encourage further interaction** by asking follow-up questions such as:
     - Preferences for specific brands, styles, or colors?
     - Any budget limitations?
   - **Refine recommendations** based on user responses.

### 6. Feedback Loop:
   - **Invite users to rate the suggestions** or provide comments to enhance future recommendations.

### 7. Tone and Style:

   - **Maintain a friendly, conversational tone** throughout.
   - Use **simple, easy-to-understand language.**
   - **Avoid jargon** and keep the conversation engaging.

**Remember:** Keep the interaction smooth and adjust based on user feedback. Your main goal is to help the user find a delightful gift and make their shopping experience enjoyable.
DO NOT PROVIDE RECOMMENDATIONS THAT ARE NOT AGE APPROPRIATE. DO NOT RECOMMEND PRODUCTS THAT ARE NOT AVAILABLE IN THE RETAILER'S CATALOG.
---

**Question:** {question}

**Context:** {context}

**History:** {history}

---

**Helpful Answer:** 

"""

# Welcome Message for the Assistant
WELCOME_MESSAGE = """**Hello and welcome!** 🎁 

I'm here to help you find the *perfect gift* for any occasion. Just tell me a bit about **who you're shopping for**, and I'll find some great ideas from our catalog. 

Let's make gift-giving **easy and fun**!

---

To get started, could you share a few details?

- Are you looking for a gift for a **child or an adult**?
- Are there any specific **interests or hobbies** the recipient enjoys?
- Is there a particular **price range** you're looking to stay within?

"""

# Constants
NB_RECOMMENDATIONS = 6  # Number of recommendations to return
LLM_MODEL_NAME = "meta-llama/Llama-Vision-Free"
EMBEDDINGS_MODEL_NAME = "togethercomputer/m2-bert-80M-2k-retrieval"
VECSTORE_PERSIST_DIRECTORY = "./chroma_vectorstore/"
DROPBOX_DIR= "/Gift_Recommendation_Bot/"
CHROMA_SUBDIR_NAME="bcbdf01a-5fe8-4eea-9e53-85e1ad37ebba"
VECTORSTORE_LINKS=[
    "chroma.sqlite3",
    "data_level0.bin",
    "header.bin",
    "index_metadata.pickle",
    "length.bin",
    "link_lists.bin"]

def download_file(local_path,dropbox_path,access_token=st.secrets["DROPBOX_ACCESS_TOKEN"]):
    # Initialize a Dropbox object using the access token
    dbx = dropbox.Dropbox(access_token)
    # Download the file to the local path
    with open(local_path, "wb") as f:
        metadata, res = dbx.files_download(path=dropbox_path)
        f.write(res.content)

# Session State Initialization for Optimized Performance---------------------------------------------

# Initialize retriever, vectorstore & embeddings model if not already in session state
if "retriever" not in st.session_state:
    embeddings = TogetherEmbeddings(
        model=EMBEDDINGS_MODEL_NAME, api_key=st.secrets["TOGETHER_API_KEY"]
    )
    if os.path.exists(VECSTORE_PERSIST_DIRECTORY):
        shutil.rmtree(VECSTORE_PERSIST_DIRECTORY)
    if not os.path.exists(VECSTORE_PERSIST_DIRECTORY):
        os.makedirs(VECSTORE_PERSIST_DIRECTORY)
    if not os.path.exists(VECSTORE_PERSIST_DIRECTORY+CHROMA_SUBDIR_NAME):
        os.makedirs(VECSTORE_PERSIST_DIRECTORY+CHROMA_SUBDIR_NAME)
    print(VECTORSTORE_LINKS[1:])
    with st.spinner("Downloading Product Catalogue..."):
        for item in VECTORSTORE_LINKS[1:]:
            download_file(VECSTORE_PERSIST_DIRECTORY+CHROMA_SUBDIR_NAME+"/"+item,DROPBOX_DIR+item)
        download_file(VECSTORE_PERSIST_DIRECTORY+VECTORSTORE_LINKS[0],DROPBOX_DIR+VECTORSTORE_LINKS[0])
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    vectorstore = Chroma(
        persist_directory=VECSTORE_PERSIST_DIRECTORY, embedding_function=embeddings
    )
    st.session_state.retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": NB_RECOMMENDATIONS}
    )

# Initialize LLM model if not already in session state
if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=st.secrets["TOGETHER_API_KEY"],
        model=LLM_MODEL_NAME,
    )

# Initialize memory for chat history if not already in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        input_key="question",
        memory_key="history",
        return_messages=True,
    )


# Define the prompt template for the QA chain
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=PROMPT,
)

# Initialize the QA chain with the language model, retriever, and memory in session state for optimized performance
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        st.session_state.llm,
        retriever=st.session_state.retriever,
        return_source_documents=True,  # Return source documents for the answer
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT,
            "verbose": True,
            "memory": st.session_state.memory,
        },
    )
# ---------------------------------------------------------------------------------------------------


# Streamlit Chat App --------------------------------------------------------------------------------
st.title("Gift Recommendation Assistant 🎁")

# Initialize messages in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": WELCOME_MESSAGE})

# Display chat messages history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and generate a response
question = st.chat_input("Enter your question:")
if question:
    # Display user question in chat
    with st.chat_message("user"):
        st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

    # Show spinner while gathering recommendations and generating response
    with st.spinner("Gathering Recommendations..."):
        result = st.session_state.qa_chain({"query": question})

    # Display assistant's response with streaming effect
    response_container = st.chat_message(
        "assistant"
    )  # Placeholder for streaming response
    response = result["result"]  # Get the response from the QA chain
    words = response.split(" ")  # Split response into words for streaming effect
    message_placeholder = (
        response_container.empty()
    )  # Empty placeholder for streaming message
    streaming_message = ""
    for i, word in enumerate(words):
        streaming_message += word + " "  # Append word to streaming message
        message_placeholder.markdown(streaming_message)  # Display streaming message
        time.sleep(0.02)  # Add a delay for streaming effect
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )  # Add response to chat history
# ---------------------------------------------------------------------------------------------------
