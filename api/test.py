import requests
from PIL import Image
import io

# Test API endpoint
def test_prediction(image_path):
    url = "http://localhost:8000/predict"
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def test_ping():
    url = "http://localhost:8000/ping"
    try:
        response = requests.get(url, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Ping Response: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("Testing /ping endpoint...")
    test_ping()
    print("\nTesting /predict endpoint...")
