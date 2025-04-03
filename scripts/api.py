import pandas as pd
import requests

# API Endpoint (Replace with your actual query)
url = "https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/DF_WITS_Tariff_TRAINS/.840.000..reported/?startperiod=2020&endperiod=2021&detail=DataOnly"


# Your WITS Login Credentials
username = "shannenkoh160403@gmail.com"  # Replace with your actual WITS email
password = "au14YFmyd"          # Replace with your WITS password

# Headers
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"  # Requesting JSON response
}

# Make API Request with Authentication
response = requests.get(url, headers=headers, auth=(username, password))

# Check if the response status is OK
if response.status_code == 200:
    try:
        # Parse the JSON response
        data = response.json()

        # Check if the 'dataSet' contains any data
        if "dataSet" in data and data["dataSet"]:
            print("Data found.")
            # Convert JSON data to a pandas DataFrame (adjust based on the structure of the JSON)
            df = pd.json_normalize(data['dataSet'])
            print(df)
        else:
            print("No data found in the response.")

    except ValueError as e:
        print(f"Error parsing JSON: {e}")
else:
    print(f"Error: {response.status_code}, Reason: {response.text}")