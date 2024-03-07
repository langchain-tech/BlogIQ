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
from prompt import get_template, sorted_keywords_string


from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

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
SERP_API_KEY = os.getenv("SERP_API_KEY")

class GraphState(TypedDict):
    keys: Dict[str, any]

def serp_api_caller(question):
    print("---Calling SerpApi---")
    params = {
        "engine": "google",
        "q": question,
        "api_key": SERP_API_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    return [details['link'] for details in organic_results]


def create_collection(collection_name, question, urls, option):
    if option == 'Use Serpi Api':
        urls = serp_api_caller(question)
    elif option == 'Use Custom Urls':
        urls
    elif option == 'Use Both of them':
        serp_urls = serp_api_caller(question)
        urls = urls + serp_urls
    print("---Got Results---")
    print(urls)
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    print(docs_list)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        embedding=OpenAIEmbeddings(),
    )
    create_record(collection_name, urls)
    print(f"Collection '{collection_name}' created successfully.")
    return vectorstore.as_retriever()

def retrieve_documents(collection_name, question):
    embedding_function = OllamaEmbeddings()
    vectorstore = Chroma(collection_name, embedding_function)
    retriever = vectorstore.as_retriever()
    documents = retriever.get_relevant_documents(question)
    return documents

def retrieve(state):
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    keywords = state_dict["keywords"]
    additional_context = state_dict["additional_context"]
    blog_words_limit = state_dict["blog_words_limit"]
    option = state_dict["option"]
    urls = state_dict["urls"]
    collection_key = secrets.token_hex(12 // 2)
    retriever = create_collection(collection_key, question, urls, option)
    documents = retriever.get_relevant_documents(question)
    return {    "keys":

                {
                    "documents": documents,
                    "question": question,
                    'keywords': keywords,
                    "additional_context": additional_context,
                    "blog_words_limit": blog_words_limit,
                    "option": option,
                    "urls": urls
                }
            }

def generate(state):
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    keywords = state_dict["keywords"]
    additional_context = state_dict["additional_context"]
    blog_words_limit = state_dict["blog_words_limit"]
    option = state_dict["option"]
    urls = state_dict["urls"]
    keywords_string = sorted_keywords_string(keywords)
    template = get_template()
    prompt = PromptTemplate(template=template, input_variables=["context", "question", "additional_context", "keywords_string", "blog_words_limit"])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.7, streaming=True, max_tokens=4096, stop=None)
    # llm = ChatOllama(model="llama2:latest")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    print(prompt)
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke(
        {
            "documents": documents,
            "question": question,
            "additional_context": additional_context,
            "keywords_string": keywords_string,
            "blog_words_limit": blog_words_limit
        }
    )
    print("------- Generated -------")
    return {    "keys":

                {
                    "documents": documents,
                    "question": question,
                    'keywords': keywords,
                    "additional_context": additional_context,
                    "blog_words_limit": blog_words_limit,
                    "option": option,
                    "urls": urls,
                    "generation": generation

                }
            }

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()



def handle_urls():
    urls_str = st.text_input("Enter URLs (separated by commas):")
    if urls_str:
        urls_str = urls_str.strip()
        if urls_str:
            urls = [url.strip() for url in urls_str.split(",")]
            return urls
        else:
            st.warning("Please enter URLs separated by commas.")
    return None

st.title("SEO Content Generator")

question = st.text_input("Enter your question:")
additional_context = st.text_area("Enter additional context:")
blog_words_limit = st.radio('Blog size in number of words:', ['500 - 1000', '1000 - 1500', '1500 - 2000', '2000 - 2500'])

keywords = {}
num_keywords = st.number_input("Number of Keywords:", min_value=1, key="num_keywords_input")
for i in range(num_keywords):
  keyword = st.text_input(f"Keyword {i+1}:", key=f"keyword_input_{i+1}")
  priority = st.number_input(f"Priority for {keyword} (higher = more important):", min_value=1, key=f"priority_input_{i+1}")
  keywords[keyword] = priority


option = st.radio('Select an option:', ['Use Serpi Api', 'Use Custom Urls', 'Use Both of them'])
if option == 'Use Serpi Api':
    st.write('Using Serp API!')
    urls = []
elif option == 'Use Custom Urls':
    st.write('Using Custom Urls!')
    urls = handle_urls()
elif option == 'Use Both of them':
    st.write('Using Both!')
    urls = handle_urls()
else:
    st.write('No option selected')

if st.button("Generate SEO Content"):
  context = {
      "question": question,
      "additional_context": additional_context,
      "keywords": keywords,
      "blog_words_limit": blog_words_limit,
      "option": option,
      "urls": urls
  }
  print(context)
  output = app.invoke({"keys": context})
  st.subheader("Generated Content:")
  st.text(output["keys"]["generation"])



#### Below code is use to manually invoke app.py

# inputs = {"keys": {
# 'question': 'How electrical tranformer works?',
# 'additional_context': 'Explore the inner workings of electrical transformers and unravel the principles of electromagnetic induction that power these crucial devices. Discover how alternating current in the primary winding generates a magnetic field, inducing voltage in the secondary winding. Delve into the transformative role of the core material, voltage relationships, and the distinction between step-up and step-down transformers. Learn how transformers facilitate efficient energy transfer in power distribution, enabling electricity to flow seamlessly across long distances and meet diverse voltage requirements. This comprehensive guide illuminates the fundamental mechanisms behind how electrical transformers work, essential for anyone seeking insights into power systems and electrical engineering.',
# 'keywords': {'Electromagnetic induction': 1}}}

# for output in app.stream(inputs):
#     for key, value in output.items():
#         pprint.pprint(f"Node '{key}':")
#     pprint.pprint("\n---\n")

# pprint.pprint(value["keys"]["generation"])




