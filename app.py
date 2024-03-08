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
    print(collection_name, question)
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
    elif step_to_execute == "Generate Content":
        collection_key = state_dict["collection_key"]
        retriever = retrieve_documents(collection_key, question)

    
    documents = retriever.get_relevant_documents(question)
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
    blog_structure = { "blog_structure_1": { "title": "The Evolution of NLP: From Language Models to ChatGPT", "headings": [ "Introduction to NLP and Language Models", "The Rise of Large Language Models (LLMs)", "Understanding the Technology Behind ChatGPT", "Training and Development of Language Models", "Applications of NLP in Real-World Scenarios", "Challenges and Limitations of Current NLP Models", "Future Trends in NLP and Language Processing", "Enhancing ChatGPT with Advanced Techniques", "The Impact of NLP on Various Industries", "Conclusion and Future Outlook" ] }, "blog_structure_2": { "title": "Unraveling the Power of NLP: A Deep Dive into Language Models", "headings": [ "Demystifying Natural Language Processing (NLP)", "Exploring the Inner Workings of Large Language Models", "The Mechanics of ChatGPT and Similar LLMs", "Insights into the Training Process of Language Models", "Practical Applications and Use Cases of NLP", "Addressing the Challenges in NLP Development", "Emerging Trends and Innovations in NLP", "Optimizing ChatGPT for Enhanced Performance", "NLP's Influence on Business and Society", "Looking Ahead: The Future of NLP and Language Understanding" ] }, "blog_structure_3": { "title": "Navigating the NLP Landscape: From LLMs to Chatbots", "headings": [ "A Primer on Natural Language Processing (NLP)", "Evolution of Language Models and LLMs", "Decoding the Inner Workings of ChatGPT", "Training Strategies for Language Models", "Real-World Applications of NLP Technology", "Overcoming Challenges in NLP Implementation", "Future Directions in NLP Research", "Advanced Techniques for Improving ChatGPT", "NLP's Impact on Diverse Industries", "Final Thoughts: Shaping the Future of NLP" ] } }
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
        template = get_structure_template()
        prompt = PromptTemplate(template=template, input_variables=["documents", "question", "additional_context", "primary_keyword", "blog_structure"])
    elif step_to_execute == "Generate Content":
        heading = state_dict["heading"]
        print("dasdasdasdas", heading)
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
    st.title("Primary Details:")

    question = st.text_input("Enter your topic:", session_data['question'])
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
    print(session_data)

    return question, primary_keyword, blog_words_limit, session_data['urls']

def generate_structure_form(session_data):
    st.title("Generate Structure:")
    additional_context = st.text_area("Enter additional context for Structure:", session_data['additional_context'])
    session_data['additional_context'] = additional_context

def main():
    st.sidebar.title("Form Wizard")

    # Initialize session_data dictionary
    if 'session_data' not in st.session_state:
        st.session_state.session_data = initialize_session_data()

    current_step = st.sidebar.selectbox("Step", ["Primary Details", "Generate Structure", "Generate Content"])

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
            st.text(temp_structure['structure'])

    elif current_step == "Generate Content":
        st.session_state.session_data['step_to_execute'] = current_step
        context = st.session_state.session_data
        structure_text = context['structure']
        parsed_structure = ast.literal_eval(structure_text)
        headings = parsed_structure['blog_structure_1']['headings']
        print("------------------------------================================",headings)
        blog_content = ""
        for heading in headings:
            context['heading'] = heading
            time.sleep(10)
            output = app.invoke({"keys": context})
            current_heading_content = output["keys"]["blog"]
            blog_content += f"## {heading}\n\n{current_heading_content}\n\n"

        blog_content

            # if st.sidebar.button("Reset", key="reset"):
            #     st.session_state.session_data = initialize_session_data()

if __name__ == "__main__":
    main()











# Initialize session_state
# if 'session_state' not in st.session_state:
#     st.session_state.session_data = {
#         'question': "",
#         'primary_keyword': "",
#         'blog_words_limit': "",
#         'urls': []
#     }

# def handle_urls():
#     urls_str = st.text_input("Enter URLs (separated by commas):")
#     if urls_str:
#         urls_str = urls_str.strip()
#         if urls_str:
#             urls = [url.strip() for url in urls_str.split(",")]
#             return urls
#         else:
#             st.warning("Please enter URLs separated by commas.")
#     return []

# def primary_details():
#     st.title("Primary Details:")
#     session_data = st.session_state.session_data
    
#     question = st.text_input("Enter your topic:", session_data['question'])
#     primary_keyword = st.text_input("Enter primary keyword:", session_data['primary_keyword'])
#     blog_words_limit_options = ['500 - 1000', '1000 - 1500', '1500 - 2000', '2000 - 2500']
#     blog_words_limit_index = blog_words_limit_options.index(session_data['blog_words_limit'] or '500 - 1000')

#     blog_words_limit = st.radio('Blog size in number of words:', blog_words_limit_options, index=blog_words_limit_index)
    
#     option = st.radio('Select an option:', ['Use Serpi Api', 'Use Custom Urls', 'Use Both of them'])
    
#     if option == 'Use Serpi Api' and question:
#         st.write('Using Serp API!')
#         session_data['urls'] = serp_api_caller(question)
#         st.write(session_data['urls'])
#     elif option == 'Use Custom Urls':
#         st.write('Using Custom Urls!')
#         session_data['urls'] = handle_urls()
#         st.write(session_data['urls'])
#     elif option == 'Use Both of them' and question:
#         st.write('Using Both!')
#         session_data['urls'] = handle_urls() + serp_api_caller(question)
#         st.write(session_data['urls'])
#     else:
#         st.write('No option selected')
    
#     # Update session_state
#     session_data['question'] = question
#     session_data['primary_keyword'] = primary_keyword
#     session_data['blog_words_limit'] = blog_words_limit
#     print(session_data)

#     return question, primary_keyword, blog_words_limit, session_data['urls']

# def generate_structure():
#     st.title("Generate Structure:")
#     additional_context = st.text_area("Enter additional context for Structure:")
    
#     return additional_context

# def main():
#     st.sidebar.title("Form Wizard")
#     current_step = st.sidebar.selectbox("Step", ["Primary Details", "Generate Structure"])

#     if current_step == "Primary Details":
#         question, primary_keyword, blog_words_limit, urls = primary_details()   
    
#     elif current_step == "Generate Structure":
#         additional_context = generate_structure()
    
#     if st.sidebar.button("Reset", key="reset"):
#         st.session_state.session_data = {
#             'question': "",
#             'primary_keyword': "",
#             'blog_words_limit': "",
#             'urls': []
#         }

# if __name__ == "__main__":
#     main()



































# st.title("SEO Content Generator")

# question = st.text_input("Enter your topic:")
# additional_context = st.text_area("Enter additional context:")
# blog_words_limit = st.radio('Blog size in number of words:', ['500 - 1000', '1000 - 1500', '1500 - 2000', '2000 - 2500'])

# # keywords = {}
# # num_keywords = st.number_input("Number of Keywords:", min_value=1, key="num_keywords_input")
# # for i in range(num_keywords):
# #   keyword = st.text_input(f"Keyword {i+1}:", key=f"keyword_input_{i+1}")
# #   priority = st.number_input(f"Priority for {keyword} (higher = more important):", min_value=1, key=f"priority_input_{i+1}")
# #   keywords[keyword] = priority
# primary_keyword = st.text_input("Enter primary keyword:")



# if st.button("Generate SEO Content"):
#   context = {
#       "question": question,
#       "additional_context": additional_context,
#       "primary_keyword": primary_keyword,
#       "blog_words_limit": blog_words_limit,
#       "option": option,
#       "urls": urls
#   }
#   print(context)
#   output = app.invoke({"keys": context})
#   st.subheader("Generated Content:")
#   st.text(output["keys"]["generation"])



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




