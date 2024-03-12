import os
import json
import pprint
import secrets
import operator
import streamlit as st
from langchain import hub
from serpapi import GoogleSearch
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict, Dict
from langgraph.graph import END, StateGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from postgres import create_record, update_record
from prompt import get_structure_template, get_content_generator_template


from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

## app imports
from st_frontend.frontend import main
from prompts.content_prompt import content_template
from prompts.structure_prompt import structure_template

### Uncomment import 'pdb' this to use debugger in the app
### Use this code in between any file or function to stop debugger at any point pdb.set_trace()
import pdb

## Used to load .env file
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

class GraphState(TypedDict):
    keys: Dict[str, any]

def create_collection(collection_name, question, urls):
    print("---Got Results---")
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print("---CREATING NEW DOCUMENTS---")
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        embedding=OpenAIEmbeddings(),
    )
    create_record(collection_name, urls)
    print(f"Collection '{collection_name}' created successfully.")
    return vectorstore.as_retriever()

def retrieve_documents(collection_name, question):
    print("---RETRIEVING OLD DOCUMENTS---")
    embedding_function = OpenAIEmbeddings()
    vectorstore = Chroma(collection_name, embedding_function)
    return vectorstore.as_retriever()
     

def retrieve(state):
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    primary_keyword = state_dict["primary_keyword"]
    additional_context = state_dict["additional_context"]
    blog_words_limit = state_dict["blog_words_limit"]
    urls = state_dict["urls"]
    step_to_execute = state_dict["step_to_execute"]

    if 'structure' in state_dict:
        structure = state_dict["structure"]
    else:
        structure = ""

    if 'heading' in state_dict:
        heading = state_dict["heading"]
    else:
        heading = ""

    if step_to_execute == "Generate Structure":
        collection_key = secrets.token_hex(12 // 2)
        retriever = create_collection(collection_key, question, urls)
        documents = retriever.get_relevant_documents(question)
    elif step_to_execute == "Generate Content":
        collection_key = state_dict["collection_key"]
        retriever = retrieve_documents(collection_key, heading)
        documents = retriever.get_relevant_documents(heading)

    return  {    "keys":

                {
                    "documents": documents,
                    "question": question,
                    'primary_keyword': primary_keyword,
                    "additional_context": additional_context,
                    "blog_words_limit": blog_words_limit,
                    "urls": urls,
                    "step_to_execute": step_to_execute,
                    "structure": structure,
                    "collection_key": collection_key,
                    "heading": heading

                }
            }

def generate(state):
    blog_structure = {
        "Blog_Structure_1":
            {
                "title": "TITLE",
                "headings":
                    [
                        "HEADING 1",
                        "HEADING 2",
                        "HEADING 3",
                        "HEADING 4",
                        "HEADING 5",
                        "HEADING 6",
                        "HEADING 7",
                        "HEADING 8",
                        "HEADING 9",
                        "HEADING 10"
                    ]
            },
        "Blog_Structure_2":
            {
                "title": "TITLE",
                "headings":
                    [
                        "HEADING 1",
                        "HEADING 2",
                        "HEADING 3",
                        "HEADING 4",
                        "HEADING 5",
                        "HEADING 6",
                        "HEADING 7",
                        "HEADING 8",
                        "HEADING 9",
                        "HEADING 10"
                    ]
            },
        "Blog_Structure_3":
            {
                "title": "TITLE",
                "headings":
                    [
                        "HEADING 1",
                        "HEADING 2",
                        "HEADING 3",
                        "HEADING 4",
                        "HEADING 5",
                        "HEADING 6",
                        "HEADING 7",
                        "HEADING 8",
                        "HEADING 9",
                        "HEADING 10"
                    ]
            }
    }
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    primary_keyword = state_dict["primary_keyword"]
    additional_context = state_dict["additional_context"]
    blog_words_limit = state_dict["blog_words_limit"]
    urls = state_dict["urls"]
    collection_key = state_dict["collection_key"]
    step_to_execute = state_dict["step_to_execute"]
    structure = state_dict["structure"]

    if step_to_execute == "Generate Structure":
        heading = ''
        template = structure_template()
        prompt = PromptTemplate(template=template, input_variables=["documents", "question", "additional_context", "primary_keyword", "blog_structure"])
    elif step_to_execute == "Generate Content":
        heading = state_dict["heading"]
        template = content_template()
        prompt = PromptTemplate(template=template, input_variables=["documents", "structure", "primary_keyword", "blog_words_limit", "refference_links", "heading"])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.7, streaming=True, max_tokens=4096)
    # llm = ChatOllama(model="llama2:latest")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = prompt | llm | StrOutputParser()

    if step_to_execute == "Generate Structure":
        generation = rag_chain.invoke(
            {
                "documents": documents,
                "question": question,
                "additional_context": additional_context,
                "primary_keyword": primary_keyword,
                "blog_words_limit": blog_words_limit,
                "refference_links": urls,
                "blog_structure": blog_structure
            }
        )
        print("------- Structure Generated -------")

    elif step_to_execute == "Generate Content":
        generation = rag_chain.invoke(
            {
                "documents": documents,
                "primary_keyword": primary_keyword,
                "blog_words_limit": blog_words_limit,
                "refference_links": urls,
                "structure": structure,
                "heading": heading
            }
        ) 
        print("------- Content Generated -------")

    return  {    "keys":

                {
                    "documents": documents,
                    "question": question,
                    'primary_keyword': primary_keyword,
                    "additional_context": additional_context,
                    "blog_words_limit": blog_words_limit,
                    "urls": urls,
                    "generation": generation,
                    "step_to_execute": step_to_execute,
                    "blog": generation,
                    "collection_key": collection_key,
                    "heading": heading

                }
            }

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

if __name__ == "__main__":
    main(app)
