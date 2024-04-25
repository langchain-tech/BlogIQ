### Uncomment import 'pdb' this to use debugger in the app
### Use this code in between any file or function to stop debugger at any point pdb.set_trace()
import pdb

def content_template(blog):
  """
  This function generates a formatted prompt for the LLM to create blog content.

  Args:
    blog (dict): Dictionary containing previously generated content (title, headings, content)
                  if any.
    heading (str): Current heading for which content needs to be generated.

  Returns:
    str: The formatted prompt template.
  """

  template = """
  **NOTE: I NEED 100 PERCENT PLAGIARISM-FREE CONTENT. DO NOT DIRECTLY COPY AND PASTE FROM KNOWLEDGE PROVIDED BELOW.**

  {blog_prompt}

  You are a world-class writer of blogs and articles named Michael. You are going to provide headings of blog one by one and you need to write engaging and informative content for the following heading:

  **You need to generate content on total {total_headings} for this blog.**
  **Currently you are going to generate content for {current_heading} heading.**

  <h2>{heading}</h2>

  **NOTE: Only add `Conclusion` and `Final Thoughts` in last heading of blog if required.**

  **Target Word Count:** {number_of_words_per_heading}

  **NOTE: Please provide content in proper <h3>, <h4>, <p>, <ul>, <li>, <ol>, <b>, <i>, <a> and other required html tags.**

  **Additionally, to further enhance user comprehension, consider including relevant code snippets within the content, especially for technical concepts. You can provide reference links for the important terms.**

  **Knowledge Base:**

  {documents}  **Important Note:** Aim for at least 200 words while maintaining clarity and avoiding excessive subheadings.

  **SEO Keywords:** {selected_keywords}

  """
  print(template + content(blog))
  return template + content(blog)


def content(blog):
  """
  This function retrieves previously generated content for the next heading.

  Args:
    blog (dict): Dictionary containing previously generated content.

  Returns:
    str: The previously generated content or an empty string.
  """

  if blog:
    return "\n**Previously Generated Content:** --> {blog_content}\n\n MOST IMPORTANT WARNING: First check the previously generated content and only than generate new content which is unique and no duplicacy happens in the content at any cost."
  else:
    return "\n"