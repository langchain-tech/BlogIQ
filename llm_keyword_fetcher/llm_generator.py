import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json  # For parsing JSON output

### Uncomment import 'pdb' this to use debugger in the app
### Use this code in between any file or function to stop debugger at any point pdb.set_trace()
import pdb  # Optional debugger

## Used to load .env file
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Create output parser and LLM instances
LLM = ChatOpenAI()


def call_llm(topic, primary_keywords):
    """
    This function calls the LLM to generate SEO-friendly meta keywords
    as a Python list for the provided topic and primary keyword.

    Args:
        topic (str): Topic of the blog.
        primary_keywords (str): Primary keyword for the blog.

    Returns:
        list: List of generated SEO-friendly meta keywords.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a world SEO enhancing engineer. Provide SEO-friendly meta keywords as a JSON object containing a 'keywords' key with a list of keywords, approximately 30-40 in total."),  # Modified prompt
        ("user", "{input}")
    ])
    chain = prompt | LLM | JsonOutputParser()

    response = chain.invoke({"input": f"Here is the topic of blog --> {topic} and primary keyword for blog --> {primary_keywords}"})
    print(response)

    try:
        return response["keywords"]
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        return []