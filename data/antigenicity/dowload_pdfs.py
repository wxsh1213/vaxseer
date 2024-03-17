import os, sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def extract_links(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all <a> tags (hyperlinks) in the parsed HTML
            links = soup.find_all('a')
            
            # Extract the href attribute from each <a> tag
            hrefs = [link.get('href') for link in links]
            
            return hrefs
        else:
            print("Failed to retrieve the webpage. Status code:", response.status_code)
            return None
    except Exception as e:
        print("An error occurred:", str(e))
        return None

def download_content(url, folder):
    try:
        # Create a folder for downloaded content if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Send a GET request to download the content
        response = requests.get(url)
        
        # Extract filename from the URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # Save the content to a file in the specified folder
        with open(os.path.join(folder, filename), 'wb') as f:
            f.write(response.content)
        print("Downloaded:", filename)
    except Exception as e:
        print("Failed to download content from", url, "Error:", str(e))


if __name__ == "__main__":
    # url = input("Enter the URL: ")
    url = "https://www.crick.ac.uk/research/platforms-and-facilities/worldwide-influenza-centre/annual-and-interim-reports"

    output_dir=sys.argv[1]

    end_date="2023-03" # feel free to change this

    extracted_links = extract_links(url)
    if extracted_links:
        for link in extracted_links:
            if link and link.endswith(".pdf"):
                reported_date = link.split("/")[-2]
                if int(reported_date.split("-")[0]) < int(end_date.split("-")[0]) or (int(reported_date.split("-")[0]) == int(end_date.split("-")[0]) and int(reported_date.split("-")[1]) <= int(end_date.split("-")[1])):
                    # print("Report:", link)
                    download_content(link, output_dir)
