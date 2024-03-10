from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain import characterTextSplitter, HuggingFaceEmbeddings, FAISS, HuggingFaceHub, load_qa_chain
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from lanchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

def add_vertical_space(space):
    for _ in range(space):
        st.write('')

with st.sidebar:
    st.title("LLM PDFChat Application")
    st.markdown('''
    ## About
    This Application is a LLM-based PDF to Question Answer built using:
    - [Streamlit] (https://streamlit.io/)
    - [Langchain] (https;//www.langchain.com/)
    - [huggingface] (https://huggingface.com/) LLM Model
    ''')
    add_vertical_space(5)
    st.write("Created by Rohit - 1302642")

def main():
    load_dotenv()
    st.header("Ask Questions From Your PDF")
    pdf = st.file_uploader("Upload here your pdf",type="pdf")

    if pdf is not None:
        st.write("File uploaded successfully")
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()

        #Split into chuncks
        text_splitter = characterTextSplitter(
            seperator = "\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            lenght_function = len
        )
        chunks = text_splitter.split_text(text)

        #Create embedding
        embeddings = HuggingFaceEmbeddings()
        st.write("Vector Embedding in progress wait a minute...")
        knowledge_base = FAISS.from_texts(chunks,embeddings) #Create FAISS vectorstore
        st.write("Vector embedding completed")
        question = st.text_input("Ask question abour your PDF: ") #Create Q&A prompt

        if question:
            docs = knowledge_base.similarity_search(question) #Create vector for similarity search
            llm = HuggingFaceHub(repo_id = "google/flan-t5-large", model_kwargs={"temperature":0.5, "max_len":1024})
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents = docs, que = question)
            st.write("Answer:",response)
            st.write("-"*100)
            st.write("Hope you got the answer!")
            st.write("Have a nice day!")
            st.write("Thank you!")
            #st.write(chunks)

if __name__ == '__main__':
    main()
