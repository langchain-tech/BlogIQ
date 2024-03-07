from langchain import PromptTemplate

# def get_template():
#     template = """
#     Provide me Good Blog
#     Note --> Generates content without imposing a maximum token limit. Must contain {blog_words_limit} words.
#     As you are one of the best content writers in the world, your name is Joe. Today, we're tasked with writing a blog post on 
#     """
#     return template

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

#     Here is the documents (knowledge base) that you need to utilise to write blog --> {documents}, meta keywords --> {keywords_string} & here is addtional context ---> {additional_context}
#     """
#     return template

def get_template():
    template = """
        Provide a comprehensive blog post on the topic of Natural Language Processing (NLP).
        
        Historical Roots of NLP:
        Discuss the evolution of NLP from its inception, highlighting key milestones and early challenges. Consider the Turing Test, early attempts, and notable achievements.

        Core Mechanisms of NLP:
        Explore fundamental NLP mechanisms like tokenization, part-of-speech tagging, named entity recognition, and sentiment analysis. Elaborate on their significance and applications.

        Evolution of NLP:
        Trace the evolution from rule-based systems to machine learning. Discuss the impact of machine learning on NLP advancements, with a focus on key models and breakthroughs.

        Contemporary NLP:
        Delve into the current era dominated by deep learning and neural networks. Highlight models such as BERT and GPT, emphasizing their contextual understanding and generative capabilities.

        Evaluation Metrics in NLP:
        Explain the metrics used to assess NLP model performance. Include insights into text classification metrics, language generation evaluation, and the importance of accurate assessment.

        Ethical Dimensions of NLP:
        Address the ethical considerations in NLP, including issues of bias, data privacy, and responsible AI usage. Discuss the challenges and proposed solutions for ethical NLP development.

        Future Trajectories of NLP:
        Look ahead to the future of NLP, considering advancements in quantum computing, integration with other AI domains, and the importance of explainable AI.

        Conclusion:
        Summarize the significance of NLP in the technological landscape. Emphasize the transformative impact and the ongoing journey toward a more connected and intelligent future.

        Additional Context:
        {additional_context}

        Meta Keywords:
        {keywords_string}

        Knowledge Base:
        {documents}

        Blog Minimum Word Limit and make sure blog contains:
        {blog_words_limit}
    """
    return template

def sorted_keywords_string(keywords):
    sorted_keywords = dict(sorted(keywords.items(), key=lambda item: item[1], reverse=True))
    return ", ".join(sorted_keywords.keys())


