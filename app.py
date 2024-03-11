import os
import ast
import time
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
        template = get_structure_template()
        prompt = PromptTemplate(template=template, input_variables=["documents", "question", "additional_context", "primary_keyword", "blog_structure"])
    elif step_to_execute == "Generate Content":
        heading = state_dict["heading"]
        template = get_content_generator_template()
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

    
def initialize_session_data():
    return {
        'question': "",
        'primary_keyword': "",
        'blog_words_limit': "",
        'urls': [],
        'additional_context': ""
    }

def handle_urls():
    urls_str = st.text_input("Enter URLs (separated by commas):")
    if urls_str:
        urls_str = urls_str.strip()
        if urls_str:
            urls = [url.strip() for url in urls_str.split(",")]
            return urls
        else:
            st.warning("Please enter URLs separated by commas.")
    return []

def primary_details(session_data):
    st.title("Primary Details to generate a blog:")

    question = st.text_input("Enter your topic name:", session_data['question'])
    primary_keyword = st.text_input("Enter primary keyword:", session_data['primary_keyword'])
    blog_words_limit_options = ['500 - 1000', '1000 - 1500', '1500 - 2000', '2000 - 2500']
    blog_words_limit_index = blog_words_limit_options.index(session_data['blog_words_limit'] or '500 - 1000')

    blog_words_limit = st.radio('Blog size in number of words:', blog_words_limit_options, index=blog_words_limit_index)
    
    option = st.radio('Select an option:', ['Use Serpi Api', 'Use Custom Urls', 'Use Both of them'])
    
    if option == 'Use Serpi Api' and question:
        st.write('Using Serp API!')
        session_data['urls'] = serp_api_caller(question)   # Replace with actual Serpi API data
        st.write(session_data['urls'])
    elif option == 'Use Custom Urls':
        st.write('Using Custom Urls!')
        session_data['urls'] = handle_urls()
        st.write(session_data['urls'])
    elif option == 'Use Both of them' and question:
        st.write('Using Both!')
        session_data['urls'] = handle_urls() + serp_api_caller(question)  # Replace with actual Serpi API data
        st.write(session_data['urls'])
    else:
        st.write('No option selected')
    
    # Update session_state
    session_data['question'] = question
    session_data['primary_keyword'] = primary_keyword
    session_data['blog_words_limit'] = blog_words_limit

    return question, primary_keyword, blog_words_limit, session_data['urls']

def generate_structure_form(session_data):
    st.title("Generate Structure:")
    additional_context = st.text_area("Enter additional context for Structure:", session_data['additional_context'])
    session_data['additional_context'] = additional_context

def convert_to_title_case(input_string):
    words = input_string.split('_')
    capitalized_words = [word.capitalize() for word in words]
    result_string = ' '.join(capitalized_words)
    return result_string

def main():
    st.sidebar.title("Blog Generator for SEO")

    if 'session_data' not in st.session_state:
        st.session_state.session_data = initialize_session_data()

    current_step = st.sidebar.radio("Step to create a Blog:", ["Primary Details", "Generate Structure", "Generate Content"])

    if current_step == "Primary Details":
        primary_details(st.session_state.session_data)

    elif current_step == "Generate Structure":
        output = ''
        generate_structure_form(st.session_state.session_data)
        st.session_state.session_data['step_to_execute'] = current_step
        if st.button("Generate SEO Content"):
            context = st.session_state.session_data
            print("=========", context)
            output = app.invoke({"keys": context})
            st.subheader("Generated Content:")
            structure = output["keys"]["generation"]
            st.session_state.session_data['structure'] = (structure or context['structure'])

        temp_structure = st.session_state.session_data

        if temp_structure and 'structure' in temp_structure:
            if output:
                st.session_state.session_data['collection_key'] = output["keys"]["collection_key"]
            data = ast.literal_eval(temp_structure['structure'])
            for key, value in data.items():
                st.write(f"## {convert_to_title_case(key)}")
                st.write(f"## Title: {value['title']}")
                for heading in value['headings']:
                    heading

            st.write(f"### Selected Blog Structure:")
            selected_blog = st.selectbox('Select a Blog Structure', list(data.keys()))
            st.session_state.session_data['selected_blog'] = selected_blog
            st.write(f"## {data[selected_blog]['title']}")
            st.write("### Headings:")
            for heading in data[selected_blog]['headings']:
                heading


    elif current_step == "Generate Content":
        st.session_state.session_data['step_to_execute'] = current_step
        context = st.session_state.session_data
        structure_text = context['structure']
        parsed_structure = ast.literal_eval(structure_text)
        headings = parsed_structure[context['selected_blog']]['headings']
        st.write(f"## Question :--> {context['question']}")
        st.write(f"## Primary Keyword :--> {context['primary_keyword']}")
        st.write(f"## Word Limit :--> {context['blog_words_limit']} approx.")
        st.write(f"## Additional Context :--> {context['additional_context']}")
        st.write(f"## Blog Title :--> {parsed_structure[context['selected_blog']]['title']}")
        for heading in headings:
            heading

        if st.button("Generate Blog Content"):
            content = ''
            for heading in headings:
                context['heading'] = heading
                time.sleep(10)
                output = app.invoke({"keys": context})
                current_heading_content = output["keys"]["blog"]
                f"## {heading}\n\n{current_heading_content}\n\n"
                content += f"## {heading}\n\n{current_heading_content}\n\n"
                st.session_state.session_data['blog'] = content
        else:
            # context
            st.session_state.session_data['blog']
            # if st.sidebar.button("Reset", key="reset"):
            #     st.session_state.session_data = initialize_session_data()

if __name__ == "__main__":
    main()
