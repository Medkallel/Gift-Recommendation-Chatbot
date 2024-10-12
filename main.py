# main.py

import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

st.set_page_config(page_icon="🎁", page_title="Gift Recommendation Assistant")


# Vector Store
persist_directory = './chroma_vectorstore/'

if "retriever" not in st.session_state:
    # Embeddings
    embeddings_model_name = "multi-qa-mpnet-base-dot-v1"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    # Retrieval and Generation
    vectorstore = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
    st.session_state.retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})

llm = ChatOpenAI(base_url="https://api.together.xyz/v1", api_key=os.getenv("TOGETHER_API_KEY"), model="meta-llama/Llama-Vision-Free")

if "memory" not in st.session_state:
    st.session_state.memory= ConversationBufferMemory(
    input_key="question",
    memory_key="history",
    return_messages=True,
)

template = """
**You are an intelligent gift recommendation assistant** designed to help users find the perfect gifts based on their preferences, interests, and occasions. Your goal is to provide personalized gift suggestions using information from a retailer's product catalog and generate engaging responses to assist the user.

### 1. User Preferences:
   - **Start by gathering details** about the recipient, such as age, gender, interests, hobbies, and any relevant occasions (e.g., birthdays, anniversaries, holidays). This helps tailor your suggestions more effectively. 
   - **Avoid repeating** this step more than twice without providing any recommendations.

### 2. Product Retrieval:
   - **Search the retailer's catalog** to find products matching the user’s criteria. 
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
     - *"https://amazon.com/dp/(PARENT_ASIN)"*

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

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=st.session_state.retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, "verbose": True, "memory": st.session_state.memory}
)

# Streamlit Interface
st.title("Gift Recommendation Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_message="""**Hello and welcome!** 🎁 

I'm here to help you find the *perfect gift* for any occasion. Just tell me a bit about **who you're shopping for**, and I'll find some great ideas from our catalog. 

Let's make gift-giving **easy and fun**!

---

To get started, could you share a few details?

- Are you looking for a gift for a **child or an adult**?
- Are there any specific **interests or hobbies** the recipient enjoys?
- Is there a particular **price range** you're looking to stay within?

"""
    st.session_state.messages.append({"role":"assistant","content":welcome_message})

#if len(st.session_state.messages) > 1:
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Enter your question:")
if question:
    with st.chat_message("user"):
        st.markdown(question)
        st.session_state.messages.append({"role":"user","content":question})
    with st.spinner("Gathering Recommendations..."):
        result = qa_chain({"query": question})
    response_container = st.chat_message("assistant")
    response = result["result"]
    words = response.split(" ")
    message_placeholder = response_container.empty()
    streaming_message=""
    for i,word in enumerate(words):
        streaming_message+=word+" "
        message_placeholder.markdown(streaming_message)
        time.sleep(0.02)
    # with st.chat_message("assistant"):
    #     st.markdown(result["result"])
    st.session_state.messages.append({"role":"assistant","content":response})
    print(response)