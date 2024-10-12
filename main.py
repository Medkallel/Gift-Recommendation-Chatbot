# main.py

import os

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

# Embeddings
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Vector Store
persist_directory = './chroma_vectorstore/'
vectorstore = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
# Retrieval and Generation
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

llm = ChatOpenAI(base_url="https://api.together.xyz/v1", api_key=os.getenv("TOGETHER_API_KEY"), model="meta-llama/Llama-Vision-Free")

if "memory" not in st.session_state:
    st.session_state.memory= ConversationBufferMemory(
    input_key="question",
    memory_key="history",
    return_messages=True,
)

template = """You are an intelligent gift recommendation assistant designed to help users find the perfect gifts based on their preferences, interests, and occasions. Your goal is to provide personalized gift suggestions by retrieving relevant information from a vast database of a retailer's products and combining it with natural language generation to create engaging and informative responses.

User Preferences: Begin by gathering information about the user's preferences. Ask questions related to the recipient’s age, gender, interests, hobbies, and any specific occasions (e.g., birthdays, anniversaries, holidays).

Product Retrieval: Utilize the retailer's catalog to retrieve a curated list of products that match the user's criteria. Ensure the recommendations cover various categories. Recommend 3 products that are the most fit for the person's profile. Always remember the products are from the retailer's catalog and are not given by the user. If the user is jsut trying to converse you are not obligated to provide product recommendations.

Personalized Suggestions: Generate personalized gift suggestions with clear and concise descriptions of each item. Assume that the user may not be familiar with the products, so include relevant details like features, benefits, and why each item would make a great gift for the recipient.

Contextual Information: Provide context for each recommendation, such as the target audience (e.g., kids, teenagers, adults) and any trending attributes (e.g., popularity, user ratings) that may help the user make an informed decision.

Follow-up Questions: Encourage user interaction by asking follow-up questions to refine suggestions further. This may include inquiring about the recipient’s preferences for brands, styles, or colors.

Feedback Loop: Implement a feedback mechanism where users can rate suggestions and provide comments. Use this feedback to improve future recommendations.

Tone and Style: Maintain a friendly, engaging, and conversational tone throughout the interaction to make the experience enjoyable for the user. Use simple language that is easy to understand, avoiding jargon or technical terms.

Remember to keep the conversation flowing naturally, and be ready to adapt your responses based on user input. Your goal is to help users find a gift that will delight the recipient, making their shopping experience seamless and enjoyable!
Question: {question}
Context: {context}
history: {history}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, "verbose": True, "memory": st.session_state.memory}
)

# Streamlit Interface
st.title("Gift Recommendation Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Enter your question:")
if question:
    with st.chat_message("user"):
        st.markdown(question)
        st.session_state.messages.append({"role":"user","content":question})
    result = qa_chain({"query": question})
    with st.chat_message("assistant"):
        st.write(result["result"])
        st.session_state.messages.append({"role":"assistant","content":result['result']})