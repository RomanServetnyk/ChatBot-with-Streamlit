import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
import bcrypt
import re

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Database connection
def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="chat"
    )

# Function to create the users table if it doesn't exist
def create_users_table():
    query = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL
    );
    """
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
    except Error as e:
        st.error(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

create_users_table()

def create_user(email, password):
    try:
        conn = create_connection()
        cursor = conn.cursor()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (email, hashed_password))
        conn.commit()
    except Error as e:
        st.error(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

def authenticate_user(email, password):
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE email = %s", (email,))
        record = cursor.fetchone()
        if record and bcrypt.checkpw(password.encode('utf-8'), record[0].encode('utf-8')):
            return True
        else:
            return False
    except Error as e:
        st.error(f"Error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

# New function to get user ID based on email        
def get_user_id(email):
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        record = cursor.fetchone()
        if record:
            return record[0]
        else:
            return None
    except Error as e:
        st.error(f"Error: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

# Email validation using regex
def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

# Password validation to ensure it meets the criteria
def is_valid_password(password):
    if len(password) < 8:
        return False
    if not re.search(r"[A-Za-z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True

# Read all PDF files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# Get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response

def signup_page():
    st.title("Sign Up")
    new_email = st.text_input("Email")
    new_password = st.text_input("New Password", type='password')
    if st.button("Sign Up"):
        if not is_valid_email(new_email):
            st.error("Invalid email format")
        elif not is_valid_password(new_password):
            st.error("Password must be at least 8 characters long, contain a letter, a number, and a special character")
        else:
            create_user(new_email, new_password)
            st.success("User created successfully!")
            st.session_state['page'] = "login"
            st.experimental_rerun()
    if st.button("Go to Sign In"):
        st.session_state['page'] = "login"
        st.experimental_rerun()

def login_page():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(email, password):
            user_id = get_user_id(email)  # Get the user ID
            st.session_state["logged_in"] = True
            st.session_state["email"] = email
            st.session_state["user_id"] = user_id
            st.experimental_rerun()
        else:
            st.error("Invalid email or password")
    if st.button("Go to Sign Up"):
        st.session_state['page'] = 'signup'
        st.experimental_rerun()

def logout():
    st.session_state.clear()

def main_content():
    st.set_page_config(
        page_title="Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
    print(st.session_state)
    # Main content area for displaying chat messages
    st.title(f"Chat with PDF files 🤖,{st.session_state['user_id']}!")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    st.sidebar.button('Logout', on_click=logout)
    # Chat input
    # Placeholder for chat messages
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

def main():
    """Main function to execute the Streamlit app."""
    # Creating session state variables if they're not already created
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state['page'] = 'login'

    if st.session_state['logged_in']:
        main_content()
    else:
        if st.session_state['page'] == 'login':
            login_page()
        else:
            signup_page()


if __name__ == "__main__":
    main()
