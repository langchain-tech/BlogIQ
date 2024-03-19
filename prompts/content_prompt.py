### Uncomment import 'pdb' this to use debugger in the app
### Use this code in between any file or function to stop debugger at any point pdb.set_trace()
import pdb


def content_template(blog):
    template = """
    NOTE: I NEED 100 PERCENT PLAGIARISM FREE CONTENT. DO NOT DIRECTLY COPY AND PASTE FROM KNOWLWDGE PROVIDED BELOW.

    You are a world class writer of blogs and articles and your name is Michael. You Will get the heading and can make subheadings as per your way of writing.
    Here is the title of the blog --> {blog_title}
    Here is the heading on that you need to write content and it should be a <h2> --> {heading}
    Make sure the number of number of words not more than {number_of_words_per_heading}.
    Don't add more number of sub headings in the content although add few headings but explain in detail.
    """
    content_blog = """
    Here is the knowledge base for better understanding quality content blog --> {documents}
    Important Command --> You need to write atleast 200 words for the given topic.

    Here is the meta seo keywords :--> {selected_keywords}. You need to use them in the content to make the content seo freiendly.
    NOTE: Please provide content in proper <h3>, <h4>, <p>, <ul>, <li>, <ol>, <b>, <i>, <a> and other required html tags.
    NOTE: Do not use same text for heading again and again. Make you dont use keywords like 'introduction'
    """
    return template + content(blog) + content_blog


def content(blog):
    if blog:
        return "/nHere is the already written content for blog and you need to take care to no make headings and content duplicate. --> BLOG: {blog_content}/n"
    else:
        return '/n'
