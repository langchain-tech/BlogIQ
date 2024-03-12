from googleads import adwords
from datetime import datetime, timedelta
import requests


### https://github.com/googleads/google-ads-python/blob/bcee4d08df0ea037d695d1bbcb595d7ee8adf9cd/examples/planning/generate_keyword_ideas.py

def get_country_code(country_name):
    url = "https://restcountries.com/v3.1/name/" + country_name
    response = requests.get(url)
    if response.status_code == 200:
        country_data = response.json()
        if len(country_data) > 0:
            return country_data[0]['cca2']
    return None

def get_google_keyword_data(phrase, start_date=None, end_date=None, country_name='United States', language='en', max_results=10):
    country_code = get_country_code(country_name)
    if country_code is None:
        print(f"Could not find country code for '{country_name}'")
        return

    location = country_code.lower()  # Use ISO 3166-1 alpha-2 country code

    # Set default start and end dates if not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')

    # Initialize AdWords client
    adwords_client = adwords.AdWordsClient.LoadFromStorage('../blogiq-416613-bdcafee958ac.json')

    # Initialize selector
    selector = {
        'searchParameters': [
            {
                'xsi_type': 'RelatedToQuerySearchParameter',
                'queries': [phrase]
            }
        ],
        'ideaType': 'KEYWORD',
        'requestType': 'IDEAS'
    }

    # Set language and location
    if language:
        selector['language'] = language
    if location:
        selector['location'] = location

    # Set date range
    date_range = {
        'min': start_date,
        'max': end_date
    }
    selector['searchParameters'].append({
        'xsi_type': 'DateRangeSearchParameter',
        'min': date_range['min'],
        'max': date_range['max']
    })

    # Set selector paging
    selector['paging'] = {
        'startIndex': '0',
        'numberResults': str(max_results)
    }

    # Get keywords
    service = adwords_client.GetService('TargetingIdeaService', version='v201809')
    result = service.get(selector)

    # Print results
    for idea in result['entries']:
        keyword = idea['data'][0]['value']['value']
        avg_monthly_searches = idea['data'][1]['value']['value']
        competition = idea['data'][6]['value']['value']
        
        # Additional fields
        ad_share = idea['data'][10]['value']['value']
        suggested_bid = idea['data'][11]['value']['value']
        search_volume_trend = idea['data'][12]['value']['value']
        average_cpc = idea['data'][14]['value']['value']
        
        print(f"Keyword: {keyword} | Avg. Monthly Searches: {avg_monthly_searches} | Competition: {competition} | Ad Share: {ad_share} | Suggested Bid: {suggested_bid} | Search Volume Trend: {search_volume_trend} | Average CPC: {average_cpc}")


if __name__ == "__main__":
    # Example usage
    phrase = "your keyword phrase"
    get_google_keyword_data(phrase)
