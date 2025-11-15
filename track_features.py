client_id = 'f448bf113f8748ada21eff77ba1dfda6'

client_secret = '141d20fe1b5145f1ac5d1dfc15506171'


import requests

# curl -X POST "https://accounts.spotify.com/api/token" \
#      -H "Content-Type: application/x-www-form-urlencoded" \
#      -d "grant_type=client_credentials&client_id=your-client-id&client_secret=your-client-secret"

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Authorization': 'Bearer your_access_token',
    'X-Custom-Header': 'MyCustomValue'
}

data = {
    "grant_type": "client_credentials",
    "client_id": client_id,
    "client_secret": client_secret
}


try:
    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, json=data)
    response_data = response.json()
    the_access_token = response_data['access_token']
