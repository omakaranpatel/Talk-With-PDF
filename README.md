Koffee with PDF
"Koffee with PDF" is a Streamlit application that allows users to upload a PDF file, process its content, and ask questions about the content using a language model. The application leverages various libraries and tools to read, split, embed, and retrieve text from the PDF, and then uses a language model to answer user queries based on the extracted text.

Features
PDF Upload: Users can upload a PDF file to the application.
Text Extraction: The application reads and extracts text from the uploaded PDF.
Text Splitting: The extracted text is split into manageable chunks for processing.
Vector Embedding: The text chunks are embedded using Spacy embeddings.
Vector Storage: The embedded vectors are stored in a FAISS vector store.
Text Retrieval: The stored vectors are used to retrieve relevant text chunks based on user queries.
Language Model: A language model (ChatGroq) is used to answer user queries based on the retrieved text.
Installation
Clone the repository:

git clone <repository_url>
cd <repository_directory>
Install the required packages:

pip install -r requirements.txt
Set environment variables:

export KMP_DUPLICATE_LIB_OK=TRUE
Usage
Run the Streamlit application:

streamlit run app.py
Upload a PDF file:

Use the file uploader in the Streamlit interface to upload your PDF file.
Ask a question:

Enter your question in the text input field and click the "Find Answer" button.
The application will process the PDF, retrieve relevant text, and use the language model to provide an answer.
Code Overview
Main Components
PDF Reading: The PdfReader from PyPDF2 is used to read the uploaded PDF file.
Text Splitting: The CharacterTextSplitter from langchain.text_splitter is used to split the extracted text into chunks.
Vector Embedding: The SpacyEmbeddings from langchain_community.embeddings.spacy_embeddings is used to create vector embeddings of the text chunks.
Vector Storage: The FAISS vector store from langchain_community.vectorstores is used to store and retrieve the embedded vectors.
Language Model: The ChatGroq from langchain_groq is used as the language model to answer user queries.
Prompt Template: The ChatPromptTemplate from langchain_core.prompts is used to create the prompt for the language model.
Agent: The AgentExecutor and create_tool_calling_agent from langchain.agents are used to create and execute the agent that handles user queries.
Example Code
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq  
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st

# Streamlit app title
st.title("Koffee with PDF")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    # Reading the uploaded file
    raw_text = ""
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        raw_text += page.extract_text()

    # Create Chunks of Data
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    text_chunks = text_splitter.split_text(raw_text)

    # Vector Embedding of Chunks
    embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
    vector_storage = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_storage.save_local("faiss_db")

    # Retrieve data
    db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_reader", "It is a tool to read data from pdfs")

    # LLM
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key="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")

    # Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
                provided context just say, "answer is not available in the context", don't provide the wrong answer""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Agent
    tool = [retrieval_chain]
    agent = create_tool_calling_agent(llm, tool, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)

    # User input for question
    user_input = st.text_input("Enter your question:")

    if st.button("Find Answer"):
        response = agent_executor.invoke({"input": user_input})
        st.write(f"Input: {response['input']}")
        st.write(f"Output: {response['output']}")

        
