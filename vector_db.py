import os
import pandas as pd
import streamlit as st
from langchain.document_loaders import PyPDFLoader, DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

COHERE_API_KEY=st.secrets['COHERE_API_KEY']
cohere_embedding=CohereEmbeddings(cohere_api_key=COHERE_API_KEY,  model="embed-multilingual-v3.0")

df=pd.read_csv('Dataset/store_data - Sheet1.csv')

def df_loader(df):
    articles=DataFrameLoader(df,page_content_column="Title")
    document=articles.load()
    return document

def text_split(data):
    text_splitter=RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=150
                    )
    docs=text_splitter.split_documents(data)
    return docs

def vector_store(docs):
    chroma_database=Chroma.from_documents(
                        documents=docs,
                        embedding=cohere_embedding,
                        persist_directory='chroma_db'
                    )
    return chroma_database.as_retriever()

def main(): 
    articles=df_loader(df)
    articles_split=text_split(articles)
    
    return vector_store(articles_split)

if __name__ == "__main__":
    main()