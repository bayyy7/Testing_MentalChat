import os
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

embedding = HuggingFaceEmbeddings(model_name='firqaaa/indo-sentence-bert-base')

df=pd.read_csv('Dataset/store_data - Sheet1.csv')

def df_loader(df):
    articles=DataFrameLoader(df,page_content_column="Description")
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
    chroma_database=Chroma(
                        embedding_function=embedding,
                        persist_directory='chroma_db'
                    )
    return chroma_database.as_retriever()

def main(): 
    articles=df_loader(df)
    articles_split=text_split(articles)
    
    return vector_store(articles_split)

if __name__ == "__main__":
    main()