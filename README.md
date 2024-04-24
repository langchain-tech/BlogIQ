## BlogIQ - Content Generation using AI

### 🚀 Introduction:

BlogIQ stands as a beacon of innovation in the realm of content creation, providing bloggers with an advanced platform powered by state-of-the-art technology, including Langchain, Langgraph, and OpenAI GPT-4 models. 🌟 Seamlessly integrated with dataforseo.com API, BlogIQ offers a comprehensive suite of features tailored to streamline and optimize the blogging process.

🔍 Step 1: Topic and Keyword Selection:

At its core, BlogIQ simplifies content creation by allowing users to specify their desired topic and primary keyword. Leveraging data from the dataforseo.com API, the app conducts an extensive Google search to curate a selection of relevant URLs. Users retain complete control, with the option to manually select additional URLs or meta keywords, or opt for automated meta keyword generation using OpenAI LLM to ensure adherence to SEO best practices. 📊💡

🛠️ Step 2: Title and Structure Generation:

Powered by Langchain, Langgraph, and OpenAI GPT-4 models, BlogIQ facilitates title and structure generation with unprecedented accuracy. Users are presented with a range of options tailored to their preferences, with the added flexibility of providing input to direct the GPT-4 models during this process, ensuring that the generated content seamlessly aligns with their vision and objectives. 💭✨

📝 Step 3: Content Generation:

Expanding upon the chosen structure, BlogIQ dynamically generates content enriched with insights gleaned from the selected URLs and meta keywords. Users can further guide the content generation process by providing prompts to the GPT-4 models, fostering personalized and engaging content creation experiences. 📝✨

💬 Step 4: FAQ Generation:

In the final stage, BlogIQ completes the content creation journey by generating FAQs for the blog. By analyzing the generated content and identifying potential questions, the app automatically generates a set of FAQs, enriching the blog post with valuable additional information and enhancing its engagement potential. 🤔💬

## Getting Started

### Prerequisites

Before running the app, make sure you have the following dependencies installed:

- Python 3.x
- pip (Python package installer)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/langschain/BlogIQ.git
    ```

2. Navigate to the project directory:

    ```bash
    cd BlogIQ
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1. Create a `.env` file in the project root directory.

2. Add the following environment variables to the `.env` file:

    ```
    LANGCHAIN_TRACING_V2=your_langchain_tracing_v2_key
    LANGCHAIN_PROJECT=your_langchain_project_key
    OPENAI_API_KEY=your_openai_api_key
    LANGCHAIN_API_KEY=your_langchain_api_key
    TAVILY_API_KEY=your_tavily_api_key
    SERP_API_KEY=your_serp_api_key
    ```

    Replace `your_langchain_tracing_v2_key`, `your_langchain_project_key`, `your_openai_api_key`, `your_langchain_api_key`, `your_tavily_api_key`, and `your_serp_api_key` with your actual API keys.

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Access the app in your web browser at [http://localhost:8501](http://localhost:8501).

3. Enter your question, meta description, and keywords in the input fields.

4. Click the "Generate SEO Content" button.

5. View the generated content in the app.

## Manual Invocation

If you want to manually invoke the app using predefined inputs, uncomment the relevant section in `app.py` and follow the instructions provided in the code comments.

## Additional Information

- The app uses Langchain and OpenAI tools for document retrieval and content generation. Make sure you have valid API keys for these services.
- For debugging purposes, you can uncomment the import of `pdb` in `app.py` to use the debugger within the app.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Special thanks to Langchain and OpenAI for providing powerful tools for natural language processing.
- The app structure and components are based on the Streamlit framework.

