# SEO Content Generator

This Streamlit app is designed to generate SEO-friendly content based on user-provided questions, meta descriptions, and keywords. It leverages Langchain and OpenAI tools to retrieve relevant documents from the web, create content collections, and generate informative and optimized content.

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

