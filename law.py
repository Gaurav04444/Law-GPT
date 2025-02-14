from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
import google.generativeai as genai
import re
import os
from typing import List, Dict
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import json

load_dotenv()

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import torch


class LegalDocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, 
            chunk_overlap=200, 
            add_start_index=True
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        self.chroma_persist_dir = "chroma_db_legal"
        self.vectorstore = None
        self.initialize_vectorstore()

    def initialize_vectorstore(self):
        if os.path.exists(self.chroma_persist_dir):
            self.vectorstore = Chroma(
                persist_directory=self.chroma_persist_dir,
                embedding_function=self.embeddings
            )
        else:
            self.vectorstore = Chroma(
                persist_directory=self.chroma_persist_dir,
                embedding_function=self.embeddings
            )

    def process_json_document(self, input_file: str):
        documents = []
        with open(input_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        documents.append(Document(
                            page_content=value,
                            metadata={"source": input_file, "key": key}
                        ))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        documents.append(Document(
                            page_content=item,
                            metadata={"source": input_file}
                        ))
        return documents

    def process_document(self, input_file: str):
        try:
            documents = []
            file_extension = os.path.splitext(input_file)[1].lower()
            
            # Handle different file types
            if file_extension == '.pdf':
                loader = PyPDFLoader(input_file)
                documents.extend(loader.load())
            elif file_extension == '.txt':
                loader = TextLoader(input_file)
                documents.extend(loader.load())
            elif file_extension == '.json':
                documents.extend(self.process_json_document(input_file))
            
            if not documents:
                print(f"No documents loaded from {input_file}")
                return
            
            # Split documents
            texts = self.text_splitter.split_documents(documents)
            
            # Add to vectorstore
            self.vectorstore.add_documents(texts)
            self.vectorstore.persist()
            
            print(f"Processed {len(texts)} sections from {input_file}")
            
        except Exception as e:
            print(f"Error processing document: {e}")
            raise


def main():
    INPUT_FILE = "legal_docs.txt"
    processor = LegalDocumentProcessor()
    
    if not os.path.exists("chroma_db_legal"):
        processor.process_document(INPUT_FILE)
    
    while True:
        question = input("\nLegal Question (or 'exit'): ")
        if question.lower() == 'exit':
            break
        print("\nResponse:", processor.get_response(question))

if __name__ == "__main__":
    main()

# HuggingFace model initialization (commented out alternative)
        # model_id = "HuggingFaceH4/zephyr-7b-beta"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     device_map="auto",
        #     torch_dtype=torch.float16
        # )
        # self.pipe = pipeline(
        #     "text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     max_length=4096,
        #     temperature=0.3,
        #     top_p=0.95,
        #     device_map="auto"
        # )