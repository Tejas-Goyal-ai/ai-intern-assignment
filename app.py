# ingestion agent
# this is the function that will load the pdf file and return the text content of the pdf file


from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(uploaded_file):

    temp_path = uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_path)

    documents = loader.load()

    # add filename metadata
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name

    return documents

# this function will split the text into chunks and return a list of chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks

#this function will create a vector store from the chunks and return the vector store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
def create_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


# Retrieval Agent
def create_retriever(vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever


def retrieve_documents(retriever, query):

    retrieved_docs = retriever.invoke(query)

    return retrieved_docs



# Instantiate the ChatGroq class to create a language model for conversation.

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


# load environment variables
load_dotenv()


# initialize llm
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)



def generate_response(query, retrieved_docs):

    # combine retrieved chunks
    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # create citations
    unique_citations = set()

    for doc in retrieved_docs:

        citation = f"Source: {doc.metadata['source']} | Page: {doc.metadata['page']}"

        unique_citations.add(citation)

    citations = "\n".join(unique_citations)

    # prompt
    prompt = f"""
    You are a helpful AI assistant.

    Answer the user's question ONLY using the provided context.

    If the answer is not present in the context, say:
    "I could not find the answer in the uploaded documents."

    Context:
    {context}

    Question:
    {query}

    Provide a clear and concise answer.
    """

    # call llm
    response = llm.invoke(prompt)

    # final answer
    final_answer = f"""
{response.content}

Citations:
{citations}
"""

    return final_answer




from typing import TypedDict, List
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):

    query: str

    retrieved_docs: List[Document]

    answer: str


def retrieval_node(state):

    docs = retrieve_documents(
        retriever,
        state["query"]
    )

    return {
        "retrieved_docs": docs
    }


def response_node(state):

    answer = generate_response(
        state["query"],
        state["retrieved_docs"]
    )

    return {
        "answer": answer
    }


graph = StateGraph(AgentState)

graph.add_node("retrieval", retrieval_node)

graph.add_node("response", response_node)


graph.add_edge(START, "retrieval")

graph.add_edge("retrieval", "response")

graph.add_edge("response", END)


app = graph.compile()








# frontend application using streamlit 

import streamlit as st

# =========================================
# STREAMLIT FRONTEND
# =========================================

import re

st.set_page_config(
    page_title="Multi-Agent RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent RAG Assistant")

st.markdown(
    "Upload PDFs and ask questions from your documents."
)


# =========================================
# FILE UPLOAD
# =========================================

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)


# =========================================
# SESSION STATE
# =========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================
# DISPLAY CHAT HISTORY
# =========================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])


# =========================================
# CHAT INPUT
# =========================================

query = st.chat_input(
    "Ask a question about the uploaded PDFs..."
)


# =========================================
# MAIN LOGIC
# =========================================

if uploaded_files and query:

    # =====================================
    # SHOW USER MESSAGE
    # =====================================

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):

        st.markdown(query)


    with st.spinner("Processing PDFs and generating answer..."):

        # =====================================
        # LOAD ALL DOCUMENTS
        # =====================================

        all_documents = []

        for uploaded_file in uploaded_files:

            documents = load_pdf(uploaded_file)

            all_documents.extend(documents)


        # =====================================
        # CREATE VECTORSTORE
        # =====================================

        chunks = split_documents(
            all_documents,
            chunk_size=500,
            chunk_overlap=100
        )

        vectorstore = create_vectorstore(chunks)

        retriever = create_retriever(vectorstore)


        # =====================================
        # LANGGRAPH NODES
        # =====================================

        def retrieval_node(state):

            docs = retrieve_documents(
                retriever,
                state["query"]
            )

            return {
                "retrieved_docs": docs
            }


        def response_node(state):

            answer = generate_response(
                state["query"],
                state["retrieved_docs"]
            )

            return {
                "answer": answer,
                "retrieved_docs": state["retrieved_docs"]
            }


        # =====================================
        # BUILD GRAPH
        # =====================================

        graph = StateGraph(AgentState)

        graph.add_node("retrieval", retrieval_node)

        graph.add_node("response", response_node)

        graph.add_edge(START, "retrieval")

        graph.add_edge("retrieval", "response")

        graph.add_edge("response", END)

        app = graph.compile()


        # =====================================
        # RUN GRAPH
        # =====================================

        result = app.invoke({
            "query": query
        })

        answer = result["answer"]


    # =========================================
    # STORE ASSISTANT MESSAGE
    # =========================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


    # =========================================
    # DISPLAY ASSISTANT MESSAGE
    # =========================================

    with st.chat_message("assistant"):

        st.markdown(answer)


    # =========================================
    # SOURCE SNIPPETS
    # =========================================

    st.markdown("## 📚 Source Snippets")

    shown_sources = set()

    for doc in result["retrieved_docs"]:

        source_key = (
            f"{doc.metadata['source']}"
            f"{doc.metadata['page']}"
        )

        # avoid duplicate snippets
        if source_key in shown_sources:
            continue

        shown_sources.add(source_key)

        # clean snippet text
        clean_text = re.sub(r'\s+', ' ', doc.page_content)

        clean_text = clean_text.replace("", "")

        clean_text = clean_text.replace("􀂾", "")

        # expandable snippet card
        with st.expander(
            f"📄 {doc.metadata['source']} | Page {doc.metadata['page']}"
        ):

            st.write(clean_text[:700] + "...")