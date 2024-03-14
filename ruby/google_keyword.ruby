require 'google-api-client'
require 'googleauth'
require 'google/apis/ads_googleads_v8'
require 'google/apis/ads_googleads_v8/services/keyword_plan_service'

def setup_google_ads_service
  # Path to your service account JSON file
  service_account_file = 'path_to_your_service_account_file.json'

  # Path to your Google Ads token file
  token_file = 'google_keyword.ruby'

  # Load service account credentials
  scopes = ['https://www.googleapis.com/auth/adwords']
  credentials = Google::Auth::ServiceAccountCredentials.make_creds(
    json_key_io: File.open(service_account_file),
    scope: scopes
  )

  # Load the access token
  credentials.access_token = File.open(token_file, 'r').read

  # Initialize the Google Ads API client
  google_ads = Google::Apis::AdsGoogleadsV8::GoogleAdsService.new
  google_ads.authorization = credentials

  google_ads
end

def get_keyword_ideas(google_ads, customer_id)
  # Initialize the KeywordPlanService client
  keyword_plan_service = Google::Apis::AdsGoogleadsV8::KeywordPlanService.new
  keyword_plan_service.client_options.application_name = 'Keyword Planner Example'

  # Set the customer ID
  keyword_plan_service.client_options.header_obj['login-customer-id'] = customer_id

  # Create a keyword plan request
  keyword_plan_request = Google::Apis::AdsGoogleadsV8::GenerateKeywordIdeasRequest.new(
    geo_target_constants: [Google::Apis::AdsGoogleadsV8::KeywordPlanGeoTargetType.new(id: 2840)], # United States
    language_constants: ['en']
  )

  # Generate keyword ideas
  response = google_ads.generate_keyword_ideas(customer_id, keyword_plan_request)

  # Print the keyword ideas
  if response.keyword_ideas && !response.keyword_ideas.empty?
    puts 'Keyword ideas:'
    response.keyword_ideas.each do |idea|
      puts "#{idea.text} (Avg. Monthly Searches: #{idea.avg_monthly_searches})"
    end
  else
    puts 'No keyword ideas found.'
  end
end

# Set your Google Ads customer ID
customer_id = '602-027-5901'

# Set up the Google Ads service
google_ads = setup_google_ads_service

# Get keyword ideas
get_keyword_ideas(google_ads, customer_id)
