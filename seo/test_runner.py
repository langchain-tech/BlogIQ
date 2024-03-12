from seo.seo import get_google_keyword_data

if __name__ == "__main__":
    # Example usage
    phrase = "ai applications in healthcare"
    #start_date = "20240101"  # Set custom start date if needed
    #end_date = "20240228"    # Set custom end date if needed
    #country_name = "United States"         # Location code for USA
    language = "en"           # Language code for English
    max_results = 30          # Maximum number of results to fetch

    get_google_keyword_data(phrase, start_date, end_date, country_name, language, max_results)
