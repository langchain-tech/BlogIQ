import ast
import time
import streamlit as st
from st_frontend.st_helper import initialize_session_data, primary_details, generate_structure_form, convert_to_title_case

### Uncomment import 'pdb' this to use debugger in the app
### Use this code in between any file or function to stop debugger at any point pdb.set_trace()
import pdb

def main(app):
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
            print("=========", context)
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

            st.write(f"### Selected Blog Structure:")
            selected_blog = st.selectbox('Select a Blog Structure', list(data.keys()))
            st.session_state.session_data['selected_blog'] = selected_blog
            st.write(f"## {data[selected_blog]['title']}")
            st.write("### Headings:")
            for heading in data[selected_blog]['headings']:
                st.write(heading)


    elif current_step == "Generate Content":
        st.session_state.session_data['step_to_execute'] = current_step
        context = st.session_state.session_data
        structure_text = context['structure']
        parsed_structure = ast.literal_eval(structure_text)
        headings = parsed_structure[context['selected_blog']]['headings']
        st.write(f"## Question :--> {context['question']}")
        st.write(f"## Primary Keyword :--> {context['primary_keyword']}")
        st.write(f"## Word Limit :--> {context['blog_words_limit']} approx.")
        st.write(f"## Additional Context :--> {context['additional_context']}")
        st.write(f"## Blog Title :--> {parsed_structure[context['selected_blog']]['title']}")
        for heading in headings:
            st.write(heading)

        if st.button("Generate Blog Content"):
            content = ''
            for heading in headings:
                context['heading'] = heading
                time.sleep(20)
                output = app.invoke({"keys": context})
                current_heading_content = output["keys"]["blog"]
                st.write(f"## {heading}\n\n{current_heading_content}\n\n")
                content += f"## {heading}\n\n{current_heading_content}\n\n"
                st.session_state.session_data['blog'] = content
        else:
            # context
            st.write(st.session_state.session_data['blog'])
            # if st.sidebar.button("Reset", key="reset"):
            #     st.session_state.session_data = initialize_session_data()