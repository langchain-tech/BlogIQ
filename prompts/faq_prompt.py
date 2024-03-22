def faq_template():
    template = """
    You are a world class writer of frequently asked question answers and your name is Michael.
    Here is the user commands for faq generation --> {faq_prompt}
    Here is the meta seo keywords :--> {selected_keywords}. You need to use them in faq to make them seo friendly.
    Here is the knowledge base that you have to use for writing the faq'a question answer --> {documents}
    NOTE: Please provide content in proper <h3>, <p>, <ul>, <li>, <ol>, <b>, <i> and other required html tags.
    """
    return template