import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from collections import namedtuple

Result = namedtuple("Result", ["success", "data", "message"])

### Uncomment import 'pdb' this to use debugger in the app
### Use this code in between any file or function to stop debugger at any point pdb.set_trace()
import pdb

load_dotenv()

DATA_FOR_SEO_TOKEN = os.getenv("DATA_FOR_SEO_TOKEN")

SERP_URL = "https://api.dataforseo.com/v3/dataforseo_labs/google/related_keywords/live"
SERP_LOCATIONS_URL="https://api.dataforseo.com/v3/dataforseo_labs/locations_and_languages"
SERP_API_URL="https://api.dataforseo.com/v3/serp/google/organic/live/regular"

def get_headers():
    return {
        'Authorization': f"Basic {DATA_FOR_SEO_TOKEN}",
        'Content-Type': 'application/json'
    }

def get_locations():
    response = requests.request("GET", SERP_LOCATIONS_URL, headers=get_headers())
    response = json.loads(response.text)

    if 'tasks' in response and len(response['tasks']) > 0 and 'result' in response['tasks'][0] and len(response['tasks'][0]['result']) > 0:
        items = response['tasks'][0]['result']
        if items and len(items) > 0:
            data = {item['location_name']: item['location_code'] for item in items}
            return Result(success=True, data={"data": data}, message="Locations and languages fetched sucessfully.")
        else:
            result = Result(success=False, data={"error": {"message": "No locations and languages found."}}, message=None)
            return result
    else:
        return Result(success=False, data={"error": {"message": "Something went wrong"}}, message=None)


def get_serp_urls(question, location_code):
    payload = [
        {
            "keyword": question,
            "location_code": location_code,
            "language_code": "en",
            "depth": 15
        }
    ]
    payload = json.dumps(payload)
    response = requests.request("POST", SERP_API_URL, headers=get_headers(), data=payload)
    response = json.loads(response.text)

    if 'tasks' in response and len(response['tasks']) > 0 and 'result' in response['tasks'][0] and len(response['tasks'][0]['result']) > 0 and 'items' in response['tasks'][0]['result'][0]:
        items = response['tasks'][0]['result'][0]['items']

        if items and len(items) > 0:
            urls = [item['url'] for item in items]
            return Result(success=True, data={"data": urls}, message="Urls fetched sucessfully.")
        else:
            result = Result(success=False, data={"error": {"message": "No url found."}}, message=None)
            return result
    else:
        return Result(success=False, data={"error": {"message": "Something went wrong"}}, message=None)




def get_keywords(question, location_code):
    # payload = "[{\"keyword\":\"Ai applications in healthcare\", \"location_code\":2840, \"language_code\":\"en\", \"depth\":3, \"include_seed_keyword\":false, \"include_serp_info\":false, \"limit\":30, \"offset\":0}]"
    keyword_payload = [
        {
            "keyword": question,
            "location_code": location_code,
            "language_code": "en",
            "depth": 3,
            "include_seed_keyword": False,
            "include_serp_info": False,
            "limit": 30,
            "offset": 0
        }
    ]
    payload = json.dumps(keyword_payload)
    response = requests.request("POST", SERP_URL, headers=get_headers(), data=payload)
    response = json.loads(response.text)

    print(response)
    if 'tasks' in response and len(response['tasks']) > 0 and 'result' in response['tasks'][0] and len(response['tasks'][0]['result']) > 0 and 'items' in response['tasks'][0]['result'][0]:
        items = response['tasks'][0]['result'][0]['items']

        if items and len(items) > 0:
        	print(items)
	        selected_data = [
	            {'keyword': item['keyword_data']['keyword'],
	             'location_code': item['keyword_data']['location_code'],
	             'language_code': item['keyword_data']['language_code'],
	             'competition': item['keyword_data']['keyword_info']['competition'],
	             'competition_level': item['keyword_data']['keyword_info']['competition_level'],
	             'cpc': item['keyword_data']['keyword_info']['cpc'],
	             'search_volume': item['keyword_data']['keyword_info']['search_volume'],
	             'Select': False
	             }
	            for item in items
	        ]
	        data_pd = pd.DataFrame(selected_data)
	        return Result(success=True, data={"data": data_pd}, message="Keywords fetched sucessfully.")
        else:
            result = Result(success=False, data={"error": {"message": "No keyword found."}}, message=None)
            return result
    else:
        return Result(success=False, data={"error": {"message": "Something went wrong"}}, message=None)

