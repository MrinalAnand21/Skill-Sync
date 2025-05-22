import streamlit as st
import base64
import time

# Use cache_data instead of experimental_memo
@st.cache_data  
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("static_files/bg3.jpg")

# CSS for background image and styles
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{img}");
    background-size: cover;
}}
.custom-header {{
    font-size: 26px;  /* Font size */ 
    border-radius: 10px;  /* Rounded corners */
    text-align: center;  /* Center text */
    margin-top: 20px;  /* Space above the header */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);  /* Optional: add shadow for depth */
    font-weight: bold;
}}
.title {{
    background-color: rgba(0, 0, 0, 0.5);
    padding: 10px;
    border-radius: 10px;
    text-align: center;  /* Center the title */
    font-family: 'Candara', sans-serif;  /* Use Candara font */
    font-size: 50px;  /* Title font size */
    font-weight: bold;  /* Make font bold */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);  /* Optional: add shadow for depth */
}}
</style>
"""


st.markdown(page_bg_img, unsafe_allow_html=True)


# Check if user is logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please log in to access this page.")
    if st.button("Go to Login Page"):
        st.switch_page("LOGIN.py")
    st.stop()

#begin
sid = "tempid"
sid = st.session_state.username

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyMuPDFLoader

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    ss_id,chat_index=session_id.split("$$")
    if ss_id not in st.session_state.store:
        st.session_state.store[ss_id]={}
    if chat_index not in st.session_state.store[ss_id]:
        st.session_state.store[ss_id][chat_index] = InMemoryChatMessageHistory()
    return st.session_state.store[ss_id][chat_index]

def is_id_present(session_id: str) -> bool:
    ss_id,chat_index=session_id.split('$$')
    if ss_id not in st.session_state.store:
        return False
    return True

def is_chat_present(session_id: str) -> bool:
    ss_id,chat_index=session_id.split('$$')
    if is_id_present(session_id) and chat_index not in st.session_state.store[ss_id]:
        return False
    return True

if "config" not in st.session_state:
    st.session_state.config = {}

def model_usage(session_id):
    ss_id,chat_index=session_id.split("$$")
    from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
    #  HuggingFaceAPIToken here
    if "llm" not in st.session_state:
        st.session_state.llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            task="text-generation",
            max_new_tokens=5000,
            do_sample=False,
        )
    if "model" not in st.session_state:    
        st.session_state.model = ChatHuggingFace(llm=st.session_state.llm)

def new_chat(session_id):
    if "with_message_history" not in st.session_state:
        #st.write(type(get_session_history(session_id)))
        st.session_state.with_message_history = RunnableWithMessageHistory(st.session_state.model, get_session_history)

    if is_id_present(session_id):
        st.session_state.initial_response = st.session_state.with_message_history.invoke([SystemMessage(content='''You are an AI-powered Learning Assistant who asks users on what he want to learn and helping him with answer and you are, designed to help users with personalized learning journeys based on their technical interests, knowledge levels, and learning goals. Your will be given context from technical PDFs and articles on various subjects, such as architecture/design, Java, Python, .NET, React, Flutter, and more for which user asks You will leverage this data to generate accurate and personalized learning paths and help in clarifying queries. keep your responses short so that they are not incomplete and also summarise the whole response''')],config=st.session_state.config)
        # with st.chat_message("assistant"):
        #     st.markdown(st.session_state.initial_response.content)


########### [Custom pdf upload]

# if "embeddings" not in st.session_state:
#     st.session_state.embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
# if "index" not in st.session_state:    
#     st.session_state.index = faiss.IndexFlatL2(len(st.session_state.embeddings.embed_query("test_query")))

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
index = faiss.IndexFlatL2(len(embeddings.embed_query("test_query")))
# Define paths for storing FAISS index
INDEX_PATH = './user_faiss_index'

# Function to save FAISS index locally
def save_vector_store(vector_store):
    clear_local_files()
    vector_store.save_local(INDEX_PATH)

# Function to load FAISS index from local storage
def load_vector_store():
    return FAISS.load_local(INDEX_PATH,embeddings,allow_dangerous_deserialization=True)
import subprocess
# Function to remove local files
def clear_local_files():
    if os.path.exists(INDEX_PATH):
        # os.remove(INDEX_PATH)
        command = f'rmdir /S /Q "{INDEX_PATH}"'  # /S removes the directory and all its contents, /Q suppresses confirmation
        result = subprocess.run(command, shell=True, check=True)

# Function to get PDF text
def get_pdf_text(pdf_docs):
    texts = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    for pdf in pdf_docs:
        randnum = uuid4()
        filename = f"users_pdf/{randnum}.pdf"  
        with open(filename, "wb") as f:
            f.write(pdf.read())
        loader = PyMuPDFLoader(filename)
        doc = loader.load()
        texts += text_splitter.split_documents(doc)
    return texts

# File uploader
pdf_docs = st.sidebar.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

# Process PDFs button
if st.sidebar.button("Process"):    
    current_texts = get_pdf_text(pdf_docs)
    st.success("Success! The PDFs have been successfully loaded. Please wait as we create the vector databases.")
    # st.sidebar.write(len(current_texts))
    new_uuids = [str(uuid4()) for _ in range(len(current_texts))]
    # st.sidebar.write(len(new_uuids))
    # Create the vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(documents=current_texts, id=new_uuids)
    save_vector_store(vector_store)  # Save the vector store locally
    st.sidebar.write(f"Processed {len(current_texts)} documents.")
    vector_store.docstore = InMemoryDocstore()  # Resets the document store
    # Reset the FAISS index
    vector_store.index.reset()  # This clears the index but retains its structure
    vector_store.index_to_docstore_id = {}  # Clear the mapping of IDs to documents
    st.session_state.custom_pdf = True
# Clear local files button
if st.sidebar.button("Clear"):
    clear_local_files()
    st.sidebar.write("Local files cleared.")
    st.session_state.custom_pdf = False

# Function to handle user queries and return similarity search results
def handle_user_query(query, k=1):
    if os.path.exists(INDEX_PATH):
        vector_store = load_vector_store()
        results = vector_store.similarity_search(query, k)
        return results
    else:
        return []

##############


from pdf_vector_db import get_related_content_in_pdf
def chats(session_id):
    #st.write(st.session_state.config)
    if is_id_present(session_id):
        history = get_session_history(session_id)
        for message in history.messages:
            if message.type == "human":
                with st.chat_message("user"):
                    user_input = message.content.split('<<<###>>> user query : ')[1]
                    st.markdown(user_input)
            elif message.type == "ai":
                with st.chat_message("assistant"):
                    st.markdown(message.content)
    if prompt := st.chat_input("Enter User Input"):    
        with st.chat_message("user"):
            st.markdown(prompt)
        if not st.session_state.custom_pdf:
            from pdf_vector_db import get_related_content_in_pdf 
            retrived_text = get_related_content_in_pdf(prompt,k=2)
            context = "Context : "
            for text in retrived_text:
                context += text.page_content
            model_input = context + '<<<###>>> user query : ' + prompt
            # pass
        else:
            st.success("This response has been generated from the user-uploaded PDF.")
            retrived_text = handle_user_query(prompt,k=5)
            # retrived_text = st.session_state.vector_store_name.similarity_search("what is science ?",2)
            context = "Context : "
            for text in retrived_text:
                context += text.page_content
            model_input = context + '<<<###>>> user query : ' + prompt
        # regrex and remove rag input
        response = st.session_state.with_message_history.invoke([HumanMessage(content=model_input)],config=st.session_state.config)
        with st.chat_message("assistant"):
            st.markdown(response.content)


#end

def log_out():
    st.success(f"User: {st.session_state.username} logged out..Redirecting to the login page!!")
    st.session_state.logged_in=False
    st.session_state.username = None
    clear_local_files()
    time.sleep(2)
    st.switch_page("LOGIN.py")

# Set the application title
st.markdown('<div class="title">🎓 SkillSync 📚<br><span style="font-size: 0.5em; vertical-align: baseline;">Personalized Learning with Generative AI</span></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)  # Adds a line break
st.write(f"Welcome, {st.session_state.username}! We're excited to kick off your personalized learning journey. Together, let’s explore new horizons and unlock your full potential. Ready to dive in?")

import datetime
def sidebar():
    if sid not in st.session_state.store:
        st.session_state.store[sid] = {}
    
    st.sidebar.title("Chat History")
    # Create a new chat
    if st.sidebar.button("New Chat") or len(st.session_state.store[sid].keys()) == 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_id = f"{sid}$$Chat - {timestamp}"
        st.session_state.config = {"configurable": {"session_id": session_id}}
        get_session_history(session_id)
        model_usage(session_id)
        new_chat(session_id)

    # Display existing chats
    if st.session_state.store[sid]:
        chat_titles = {chat: chat for chat in st.session_state.store[sid].keys()}
        chat_index = st.sidebar.radio("Chats", list(chat_titles.keys()))
        session_id = f"{sid}$${chat_index}"
        st.session_state.config = {"configurable": {"session_id": session_id}}
        chats(session_id)

    st.sidebar.divider()

    # Logout button
    if st.sidebar.button("Log Out"):
        log_out()
        
sidebar()
