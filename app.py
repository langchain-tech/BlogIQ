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
from prompts.feedback_content_prompt import feedback_content_template
from prompts.faq_prompt import faq_template

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
    structure_prompt = state_dict["structure_prompt"]
    urls = state_dict["selected_urls"]
    step_to_execute = state_dict["step_to_execute"]
    selected_keywords = state_dict["selected_keywords"]

    if 'faq_prompt' in state_dict:
        faq_prompt = state_dict['faq_prompt']
    else:
        faq_prompt = ''

    if 'blog_prompt' in state_dict:
        blog_prompt = state_dict['blog_prompt']
    else:
        blog_prompt = ''

    if 'number_of_words_per_heading' in state_dict:
        number_of_words_per_heading = state_dict['number_of_words_per_heading']
    else:
        number_of_words_per_heading = ''

    if 'blog_content' in state_dict:
        blog_content = state_dict['blog_content']
    else:
        blog_content = ''

    if 'blog_title' in state_dict:
        blog_title = state_dict["blog_title"]
    else:
        blog_title = ''

    if 'blog' in state_dict:
        blog = state_dict["blog"]
    else:
        blog = ''

    if 'rephrase_context' in state_dict:
        rephrase_context = state_dict["rephrase_context"]
    else:
        rephrase_context = ''

    if 'rephrase' in state_dict:
        rephrase = state_dict["rephrase"]
    else:
        rephrase = ''

    if 'structure' in state_dict:
        structure = state_dict["structure"]
    else:
        structure = ""

    if 'heading' in state_dict:
        heading = state_dict["heading"]
    else:
        heading = ""


    if 'collection_key' in state_dict:
        collection_key = state_dict["collection_key"]
        retriever = retrieve_documents(collection_key, heading)
    else:
        collection_key = secrets.token_hex(12 // 2)
        retriever = create_collection(collection_key, question, urls)

    documents = retriever.get_relevant_documents(heading)

    return  {    "keys":

                {
                    "documents": documents,
                    "question": question,
                    'primary_keyword': primary_keyword,
                    "structure_prompt": structure_prompt,
                    "urls": urls,
                    "step_to_execute": step_to_execute,
                    "structure": structure,
                    "collection_key": collection_key,
                    "heading": heading,
                    "rephrase_context": rephrase_context,
                    "rephrase": rephrase,
                    "blog": blog,
                    "blog_title": blog_title,
                    "selected_keywords": selected_keywords,
                    "blog_content": blog_content,
                    "number_of_words_per_heading": number_of_words_per_heading,
                    "blog_prompt": blog_prompt,
                    "faq_prompt": faq_prompt

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
    structure_prompt = state_dict["structure_prompt"]
    urls = state_dict["urls"]
    collection_key = state_dict["collection_key"]
    step_to_execute = state_dict["step_to_execute"]
    structure = state_dict["structure"]
    heading = state_dict["heading"]
    rephrase_context = state_dict["rephrase_context"]
    rephrase = state_dict["rephrase"]
    blog = state_dict["blog"]
    blog_title = state_dict["blog_title"]
    selected_keywords = state_dict['selected_keywords']
    blog_content = state_dict['blog_content']
    number_of_words_per_heading = state_dict['number_of_words_per_heading']
    blog_prompt = state_dict['blog_prompt']
    faq_prompt = state_dict['faq_prompt']

    if step_to_execute == "Generate Structure":
        heading = ''
        template = structure_template()
        prompt = PromptTemplate(template=template, input_variables=["documents", "question", "structure_prompt", "primary_keyword", "blog_structure", "selected_keywords"])
    elif rephrase == True:
        template = feedback_content_template()
        prompt = PromptTemplate(template=template, input_variables=["documents", "structure", "primary_keyword", "refference_links", "rephrase_context", "blog", "structure_prompt"])
    elif step_to_execute == "Generate Blog":
        heading = state_dict["heading"]
        template = content_template(blog_content)
        prompt = PromptTemplate(template=template, input_variables=["documents", "structure", "primary_keyword", "number_of_words_per_heading", "refference_links", "heading", "blog_title", "selected_keywords", "blog_content", "blog_prompt"])
    elif step_to_execute == "Generate Faq's":
        template = faq_template()
        prompt = PromptTemplate(template=template, input_variables=["documents", "primary_keyword", "selected_keywords", "faq_prompt"])

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
                "structure_prompt": structure_prompt,
                "primary_keyword": primary_keyword,
                "refference_links": urls,
                "blog_structure": blog_structure,
                "selected_keywords": selected_keywords
            }
        )
        print("------- Structure Generated -------")

    elif rephrase == True:
        generation = rag_chain.invoke(
            {
                "documents": documents,
                "primary_keyword": primary_keyword,
                "refference_links": urls,
                "structure": structure,
                "heading": heading,
                "blog": blog,
                "blog_title": blog_title,
                "rephrase_context": rephrase_context,
                "structure_prompt": structure_prompt
            }
        )
        print("------- Content Rephrased -------")

    elif step_to_execute == "Generate Blog":
        generation = rag_chain.invoke(
            {
                "documents": documents,
                "primary_keyword": primary_keyword,
                "refference_links": urls,
                "structure": structure,
                "heading": heading,
                "blog": blog,
                "blog_title": blog_title,
                "selected_keywords": selected_keywords,
                "blog_content": blog_content,
                "number_of_words_per_heading": number_of_words_per_heading,
                "blog_prompt": blog_prompt
            }
        ) 
        print("------- Content Generated -------")

    elif step_to_execute == "Generate Faq's":
        generation = rag_chain.invoke(
            {
                "documents": documents,
                "primary_keyword": primary_keyword,
                "selected_keywords": selected_keywords,
                "faq_prompt": faq_prompt,
            }
        )
        print("------- Faq's Generated -------")

    return  {    "keys":

                {
                    "documents": documents,
                    "question": question,
                    'primary_keyword': primary_keyword,
                    "structure_prompt": structure_prompt,
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
