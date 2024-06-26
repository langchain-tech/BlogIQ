def get_structure_template():
    template = """
    Please provide me structure of blog of on {question}, must have a structure, must have good headings and must have 10 headings.
    Important Commands --> Need to provide a good title and its must contain the key word --> {primary_keyword}
    Additional commands need to follow -->
    {structure_prompt}

    Here is the knowledge base for better understanding quality content blog --> {documents}
    NOTE --> Provide at least three sets of structure with different variations in heading.
    Note to Content Creator: Please ensure that the content you generate is entirely original and not copied from any existing sources. Avoid direct copy-pasting and provide unique insights and perspectives. Plagiarism is strictly prohibited, and we require 100% unique content for the blog. Make sure to use your own words and creativity to deliver high-quality, original content. Thank you for your understanding and adherence to these guidelines.
    Very Import Note -->  Proivde structure as json
    Here is the DEMO structure to follow --> {blog_structure}
    """
    return template

# def get_content_generator_template():
#     template = """
#     You are a world class blog write and your name is Michael. You need to write a write a blog using the first object from the python dictionary.
#     Here is the strcuture dictionary object --> {structure}
#     Here is the knowledge base for better understanding quality content blog --> {documents}
#     Important Command --> You need to write atleast 200 words for each heading.
#     """
#     return template

def get_content_generator_template():
    template = """
    You are a world class writer of blogs and articles and your name is Michael. Will provide you the heading and can make subheadings as per your way of writing.
    Here is the heading on that you need to write content --> {heading}
    Here is the knowledge base for better understanding quality content blog --> {documents}
    Important Command --> You need to write atleast 200 words for the given topic.
    """
    return template


# def get_template():
#     template = """
#     Provide me Good Blog.
#     Here is the question on which you need to write a blog. Question --> {question}
#     Note --> Generates content without imposing a maximum token limit. Must contain {blog_words_limit} words.
#     As you are one of the best content writers in the world, your name is Joe. Today, we're tasked with writing a blog post, and it's essential that we adhere to the following ruleset:

#     1) Make sure each paragraph must be 100 to 150 words long.
#     2) The blog should be human-readable and unique.
#     3) A conclusion should be included at the end of the blog.
#     4) The blog should follow a common blogging structure.
#     5) Blog should in multiple paragraphs 
#     6) Use meta keywords in blog writing
#     7) Make sure not to copy anything from documents (knowledge base) directly, otherwise it will be plagiarism

#     I'll be provided with documents to review and understand the topics we'll be covering. Additionally, I'll provide meta keywords that we need to incorporate to enhance our ranking on Google search.

#     Here is the documents (knowledge base) that you need to utilise to write blog --> {documents}, primary meta keywords --> {primary_keyword} & here is addtional context ---> {structure_prompt}
#     """
#     return template

# def get_template():
#     template = """
#         Provide a comprehensive blog post on the topic of Natural Language Processing (NLP).
        
#         Additional Context:

#         {structure_prompt}

#         Meta Keywords:
#         {keywords_string}

#         Knowledge Base:
#         {documents}

#         Blog Minimum Word Limit and make sure blog contains:
#         {blog_words_limit}
#     """
#     return template




