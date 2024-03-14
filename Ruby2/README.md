# Generating Refresh Token for Google Ads API Authentication

To authenticate with the Google Ads API and obtain a refresh token, follow these steps:

## Step 1: Generate Refresh Token

 Run the following command in your terminal:

   ```bash
   ruby google_auth.rb -P google_account.json
   ```

This command initiates the process to generate a refresh token.

 1) You'll receive an output containing a URL. Open this URL in your web browser. It will redirect you to the Google Accounts page to authorize access to your account.

 2) After granting permission, you'll be redirected to a page that may not load fully. However, you'll find the refresh token in the terminal output.

 3) Copy the refresh token provided in the terminal output. It will look something like this:

   ```bash
    Your refresh token is: 1//0gghEt4-IDqccCgYIARAAGBASNwF-L9IryZI5Yg2t9BMuKt7Hfo9JCqVMl3GgRhvjKI0k6LEfSwcqeUU5ItWNMYhwtdGgX0OVa04
   ```

Paste the refresh token into your google_ads_config.rb file in your home directory. 

## Step 2: Generate Keywords

  ```bash
    ruby keyword_explorer.rb
  ```