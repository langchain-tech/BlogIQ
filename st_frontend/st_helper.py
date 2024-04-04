import streamlit as st
import pycountry
import types


## app imports
from api_helper.serp_api import serp_api_caller
from seo.data_for_seo_api import get_keywords, get_serp_urls
from llm_keyword_fetcher.llm_generator import call_llm

### Uncomment import 'pdb' this to use debugger in the app
### Use this code in between any file or function to stop debugger at any point pdb.set_trace()
import pdb

# country_names = [country.name for country in pycountry.countries]
locations = {'Algeria': 2012, 'Angola': 2024, 'Azerbaijan': 2031, 'Argentina': 2032, 'Australia': 2036, 'Austria': 2040, 'Bahrain': 2048, 'Bangladesh': 2050, 'Armenia': 2051, 'Belgium': 2056, 'Bolivia': 2068, 'Brazil': 2076, 'Bulgaria': 2100, 'Myanmar (Burma)': 2104, 'Cambodia': 2116, 'Cameroon': 2120, 'Canada': 2124, 'Sri Lanka': 2144, 'Chile': 2152, 'Taiwan': 2158, 'Colombia': 2170, 'Costa Rica': 2188, 'Croatia': 2191, 'Cyprus': 2196, 'Czechia': 2203, 'Denmark': 2208, 'Ecuador': 2218, 'El Salvador': 2222, 'Estonia': 2233, 'Finland': 2246, 'France': 2250, 'Germany': 2276, 'Ghana': 2288, 'Greece': 2300, 'Guatemala': 2320, 'Hong Kong': 2344, 'Hungary': 2348, 'India': 2356, 'Indonesia': 2360, 'Ireland': 2372, 'Israel': 2376, 'Italy': 2380, "Cote d'Ivoire": 2384, 'Japan': 2392, 'Kazakhstan': 2398, 'Jordan': 2400, 'Kenya': 2404, 'South Korea': 2410, 'Latvia': 2428, 'Lithuania': 2440, 'Malaysia': 2458, 'Malta': 2470, 'Mexico': 2484, 'Morocco': 2504, 'Netherlands': 2528, 'New Zealand': 2554, 'Nicaragua': 2558, 'Nigeria': 2566, 'Norway': 2578, 'Pakistan': 2586, 'Panama': 2591, 'Paraguay': 2600, 'Peru': 2604, 'Philippines': 2608, 'Poland': 2616, 'Portugal': 2620, 'Romania': 2642, 'Saudi Arabia': 2682, 'Senegal': 2686, 'Serbia': 2688, 'Singapore': 2702, 'Slovakia': 2703, 'Vietnam': 2704, 'Slovenia': 2705, 'South Africa': 2710, 'Spain': 2724, 'Sweden': 2752, 'Switzerland': 2756, 'Thailand': 2764, 'United Arab Emirates': 2784, 'Tunisia': 2788, 'Turkiye': 2792, 'Ukraine': 2804, 'North Macedonia': 2807, 'Egypt': 2818, 'United Kingdom': 2826, 'United States': 2840, 'Burkina Faso': 2854, 'Uruguay': 2858, 'Venezuela': 2862}
frozen_locations = types.MappingProxyType(locations)

def handle_success(result):
    print("Success:", result.data)

# Function to handle failure
def handle_failure(result):
    print("Failure:", result.data["error"]["message"])

def initialize_session_data():
    return {
        'question': "",
        'primary_keyword': "",
        'urls': [],
        'structure_prompt': "",
        'selected_meta_keywords': [],
        'secondary_keywords': [],
        'selected_keywords': [],
        'manual_keywords': [],
        'country': 'United States',
        'selected_urls': [],
        'keyword': [],
        'blog': '',
        'selected_headings': '',
        'gen_step': '',
        'blog_title': '',
        'blog_prompt': '',
        'faq_prompt': '',
        'faqs': ''
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

def handle_serp_api(option, question, session_data):
    urls = []
    if (option == 'Use Serpi Api' or option == 'Use Both of them') and question:
        if st.button("Fetch Urls from DataForSeo"):
            response = get_serp_urls(question, session_data['country'])
            urls = response.data['data']

    if option == 'Use Serpi Api' and question:
        return urls
    elif option == 'Use Custom Urls':
        return handle_urls()
    elif option == 'Use Both of them' and question:
        return (handle_urls() + urls)
    else:
        st.write('Topic Must be present!')
        return []


def primary_details(session_data):
    st.title("Primary Details For Content Generation:")

    question = st.text_input("Enter your topic name:", session_data['question'])
    primary_keyword = st.text_input("Enter primary keyword:", session_data['primary_keyword'])
    selected_country = st.selectbox("Select a country", ['United States'])
    session_data['country'] = frozen_locations[selected_country]
    option = st.radio('Select an option:', ['Use Serpi Api', 'Use Custom Urls', 'Use Both of them'])
    urls = handle_serp_api(option, question, session_data)
    if len(urls) > 0:
        session_data['urls'] = urls
    selected_urls = st.multiselect("Select Urls", session_data['urls'])
    st.write("Available urls from DataForSeo:")
    st.write(session_data['urls'])
    if selected_urls:
        session_data['selected_urls'] = selected_urls

    if st.button("Reset selected from DataForSeo:"):
        session_data['selected_urls'] = []
    st.write("Selected urls from DataForSeo:")
    st.write(session_data['selected_urls'])

    session_data['option'] = option

    if question and primary_keyword and st.button('Fetch Secondary keywords Using LLM:'):
        keywords = call_llm(question, primary_keyword)
        session_data['keywords'] = keywords

    if 'keywords' in session_data:
        st.write(f"Available keywords --> {session_data['keywords']}")

    if 'keywords' in session_data:
        # keywords_s = session_data['selected_meta_keywords']
        selected_meta_keywords = st.multiselect("Select Secondary Keywords", session_data['keywords'])
        if selected_meta_keywords:
            session_data['selected_meta_keywords'] = selected_meta_keywords
    selected_rows = ''
    if st.button("Fetch keywords from DataForSeo"):
        success_result = get_keywords(primary_keyword, frozen_locations[selected_country])
        if success_result.success:
            sec_keywords = success_result.data['data']
            session_data['sec_keywords'] = sec_keywords
            handle_success(success_result)
        else:
            handle_failure(success_result)
            st.write(f"No similar keywords found for your primary keyword on --> {primary_keyword}")

    if 'sec_keywords' in session_data:
        st.write('Select Secondary keywords:')
        data = session_data['sec_keywords'].reindex(columns=['Select', 'keyword', 'search_volume', 'competition', 'competition_level', 'cpc', 'language_code'])
        selected_rows = st.data_editor(
            data,
            num_rows="dynamic",
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

    manual_keywords = st.text_input("Enter Manual Keywords (comma separated):")
    if manual_keywords:
        manual_keywords = manual_keywords.split(',')
        session_data['manual_keywords'] = manual_keywords

    if selected_rows:
        selected_keywords = set(session_data['manual_keywords'] + selected_rows['keyword'] + list(session_data['selected_meta_keywords']) + list(session_data['selected_keywords']))
    else:
        selected_keywords = set(session_data['manual_keywords'] + list(session_data['selected_meta_keywords']) + list(session_data['selected_keywords']))

    if st.button("Reset Selected keywords"):
        if selected_rows:
            selected_rows['keyword'] = []
        session_data['selected_meta_keywords'] = []
        selected_keywords = []

    st.write(f"## Selected Meta Keywords :-->")
    st.write("<ul>", unsafe_allow_html=True)
    for keyword in selected_keywords:
        st.write(f"<li>{keyword}</li>",unsafe_allow_html=True)
    st.write("</ul>", unsafe_allow_html=True)

    session_data['selected_keywords'] = selected_keywords
    session_data['question'] = question
    session_data['primary_keyword'] = primary_keyword

    return question, primary_keyword, session_data['urls'], session_data['selected_urls']

def generate_structure_form(session_data):
    structure_prompt = st.text_area("Enter Prompt for Structure Generation:", session_data['structure_prompt'])
    session_data['structure_prompt'] = structure_prompt
    st.write(f"## Country --> {session_data['country']}")
    st.write(f"## Selected Serp Urls -->")
    st.write(session_data['selected_urls'])
    st.write(f"## Selected Meta Keywords :-->")
    st.write("<ul>", unsafe_allow_html=True)
    for keyword in session_data['selected_keywords']:
        st.write(f"<li>{keyword}</li>",unsafe_allow_html=True)
    st.write("</ul>", unsafe_allow_html=True)

def convert_to_title_case(input_string):
    words = input_string.split('_')
    capitalized_words = [word.capitalize() for word in words]
    result_string = ' '.join(capitalized_words)
    return result_string