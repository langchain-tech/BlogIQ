import ast
import time
import streamlit as st
from api_helper.ghost_api import post_blog
from st_frontend.st_helper import initialize_session_data, primary_details, generate_structure_form, convert_to_title_case

### Uncomment import 'pdb' this to use debugger in the app
### Use this code in between any file or function to stop debugger at any point pdb.set_trace()
import pdb

def average_number_of_words(range_str, num_headings):
    lower_bound, upper_bound = map(int, range_str.split(" - "))
    average_words = (lower_bound + upper_bound) // 2
    return average_words // num_headings

def main(app):
    st.set_page_config(page_title='AI Blog Generator', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.sidebar.title("Blog Generator for SEO")

    if 'session_data' not in st.session_state:
        st.session_state.session_data = initialize_session_data()

    current_step = st.sidebar.radio("Step to create a Blog:", ["Primary Details", "Generate Structure", "Generate Content"])

    if current_step == "Primary Details":
        primary_details(st.session_state.session_data)
    elif current_step == "Generate Structure":
        output = ''
        generate_structure_form(st.session_state.session_data)
        st.session_state.session_data['step_to_execute'] = current_step
        if st.button("Generate Blog Structure"):
            context = st.session_state.session_data
            output = app.invoke({"keys": context})
            st.subheader("Generated Content:")
            structure = output["keys"]["generation"]
            st.session_state.session_data['structure'] = (structure or context['structure'])

        temp_structure = st.session_state.session_data

        if temp_structure and 'structure' in temp_structure:
            if output:
                st.session_state.session_data['collection_key'] = output["keys"]["collection_key"]
            data = ast.literal_eval(temp_structure['structure'])
            for key, value in data.items():
                st.write(f"## {convert_to_title_case(key)}")
                st.write(f"## Title: {value['title']}")
                for heading in value['headings']:
                    st.write(heading)

            titles = [value['title'] for value in data.values()]
            title = st.selectbox("Select Title", titles)
            all_headings = [heading for value in data.values() for heading in value['headings']]

            manual_headings = st.text_input("Enter Manual headings (comma separated):")
            manual_headings = manual_headings.split(',')
            st.session_state.session_data['manual_headings'] = manual_headings

            if manual_headings:
                all_headings = all_headings + manual_headings


            selected_headings = st.multiselect('Select Headings', all_headings)
            if st.button("Reset Headings"):
                st.session_state.session_data['selected_headings'] = []

            if selected_headings:
                st.session_state.session_data['selected_headings'] = selected_headings
            st.session_state.session_data['blog_title'] = title
            st.write(f"### Selected Blog Structure:")
            st.write(f"## {st.session_state.session_data['blog_title']}")
            st.write("### Headings:")
            for heading in st.session_state.session_data['selected_headings']:
                st.write(heading)



    elif current_step == "Generate Content":
        st.session_state.session_data['step_to_execute'] = current_step
        context = st.session_state.session_data
        structure_text = context['structure']
        parsed_structure = ast.literal_eval(structure_text)
        headings =  context['selected_headings']
        title = context['blog_title']

        st.write(f"## Question :--> {context['question']}")
        st.write(f"## Primary Keyword :--> {context['primary_keyword']}")
        st.write(f"## Word Limit :--> {context['blog_words_limit']} approx.")
        st.write(f"## Additional Context :--> {context['additional_context']}")
        st.write(f"## Selected Meta Keywords :-->")
        st.write("<ul>", unsafe_allow_html=True)
        for keyword in context['selected_keywords']:
            st.write(f"<li>{keyword}</li>",unsafe_allow_html=True)
        st.write("</ul>", unsafe_allow_html=True)

        st.write(f"## Blog Title :--> {title}")

        for heading in headings:
            st.write(heading)


        context['number_of_words_per_heading'] = average_number_of_words(context['blog_words_limit'], len(headings))

        if st.button("Generate Blog Content"):
            context['rephrase'] = False
            content = ''
            st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
            for heading in headings:
                context['heading'] = heading
                context['blog_title'] = title
                context['blog_content'] = content
                time.sleep(20)
                output = app.invoke({"keys": context})
                current_heading_content = output["keys"]["blog"]
                content += f"{current_heading_content}\n\n"
                st.session_state.session_data['blog'] = content
                st.markdown(f"{current_heading_content}\n\n", unsafe_allow_html=True)

            re_content = st.text_area("Enter your feedback to rephrase content:", height=300)
            if re_content and st.button("Click to rephrase content"):
                context['rephrase'] = True
                context['rephrase_context'] = re_content
                context['blog'] = st.session_state.session_data['blog']
                content = app.invoke({"keys": context})

            content = st.text_area("Edit Blog Content", value=content, height=600)

            if st.button("Save Changes!!!"):
                st.session_state.session_data['blog'] = content

            if st.button("Post Blog to Blog WebiSte"):
                response = post_blog(title, content)
        else:
            st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
            st.markdown(st.session_state.session_data['blog'], unsafe_allow_html=True)

            content = st.text_area("Enter your feedback to rephrase content:", height=300)
            if content and st.button("Click to rephrase content"):
                context['rephrase'] = True
                context['rephrase_context'] = content
                context['blog'] = st.session_state.session_data['blog']
                content = app.invoke({"keys": context})
                print(content["keys"]["blog"])
                st.session_state.session_data['blog'] = content["keys"]["blog"]

            content = st.text_area("Edit Blog Content", value=st.session_state.session_data['blog'], height=600)
            if st.button("Save Changes!!!"):
                st.session_state.session_data['blog'] = content

            if st.button("Post Blog to `langchain.ca`"):
                response = post_blog(title, content)

            # if st.sidebar.button("Reset", key="reset"):
            #     st.session_state.session_data = initialize_session_data()