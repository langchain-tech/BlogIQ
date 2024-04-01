def structure_template():
    template = """
    Topic for blog --> {question}.
    Important Commands --> Need to provide a good title and its must contain this meta key word --> {primary_keyword}
    Important user commands to follow -->
    {structure_prompt}
    Here is the knowledge base for better understanding quality content blog --> {documents}
    NOTE --> Provide at least three sets of structure with different variations in heading.
    Note to Content Creator: Please ensure that the content you generate is entirely original and not copied from any existing sources. Avoid direct copy-pasting and provide unique insights and perspectives. Plagiarism is strictly prohibited, and we require 100% unique content for the blog. Make sure to use your own words and creativity to deliver high-quality, original content. Thank you for your understanding and adherence to these guidelines.
    Very Import Note -->  Proivde structure as json
    Here is the DEMO structure to follow --> {blog_structure}
    """
    return template