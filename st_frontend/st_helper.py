import streamlit as st

## app imports
from api_helper.serp_api import serp_api_caller
from seo.data_for_seo_api import get_keywords

def handle_success(result):
    print("Success:", result.data)

# Function to handle failure
def handle_failure(result):
    print("Failure:", result.data["error"]["message"])

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


    if question:
        success_result = get_keywords(question)
        if success_result.success:
            st.write('Select Secondary keywords:')
            
            selected_rows = st.data_editor(
                success_result.data['data'],
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Your Keyword?",
                        help="Select your keywords!",
                        default=False,
                    )
                },
                hide_index=True,
            )

            selected_rows = {key: [value[i] for i in range(len(value)) if selected_rows['Select'][i]] for key, value in selected_rows.items()}
            st.write('Selected keywords:')
                
            st.data_editor(
                selected_rows,
                hide_index=True,
                disabled=["Select"]
            )
            session_data['secondary_keywords'] = selected_rows
            handle_success(success_result)
        else:
            handle_failure(success_result)
            st.write(f"No keyword found for your topic on --> {question}")

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