import requests
import time

# URL de votre application Streamlit
url = "https://khelifayacine-portfolio-portfolio-yacine-khelifa-vc7igd.streamlit.app/"

while True:
    try:
        response = requests.get(url)
        print(f"Ping {url} - Status: {response.status_code}")
    except Exception as e:
        print(f"Erreur lors du ping : {e}")
    time.sleep(900)  # Ping toutes les 15 minutes
