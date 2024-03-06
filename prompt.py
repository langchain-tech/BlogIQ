from langchain import PromptTemplate

def get_template():
    template = """
    **Title:** Write a comprehensive and informative blog post answering the following question: {question}

    **Introduction:**

    In today's world, many people are curious about {question}. This blog post will delve into this topic, providing clear explanations, valuable insights, and drawing upon the latest research and information.

    **Body:**

    * **What is {question}?** (Define the core concept, incorporating insights from {documents})
    * **Why is {question} important?** (Highlight its significance, referencing {documents})
    * **How does {question} work?** (Explain the process/mechanism) (if applicable)
    * **What are the benefits of {question}?** (List and elaborate, leveraging {documents})
    * **Are there any challenges associated with {question}?** (Discuss potential drawbacks, citing {documents})
    * **What are some real-world applications of {question}?** (Provide practical examples, supported by {documents})

    **Conclusion:**

    This blog post has explored {question} in detail. By understanding {documents}, you can gain valuable knowledge about this topic.

    **Meta Description:** {meta_description}

    **Keywords:** {keywords_string}

    **Additional Resources:**

    * Consider including links to the relevant documents used within the body for further exploration.
    * Here is the documents that have latest knowlege about the question --> {documents}

    **Write the blog post in a clear, concise, and engaging style, ensuring it is informative and SEO-friendly.**
    """
    return template

def sorted_keywords_string(keywords):
    sorted_keywords = dict(sorted(keywords.items(), key=lambda item: item[1], reverse=True))
    return ", ".join(sorted_keywords.keys())


