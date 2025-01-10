import requests

PAGE_ACCESS_TOKEN = "EAAINTHSVGO8BO8Im4D7840QmdYPSCyNYbGwKodwWCkRQhNkKrQld0F8hAq2NWhnY1i0N4ZCaLuRhebpT1aHAI8tsRd5Cr0ExrZCUpNAb8oSKxeqp7J9ZBJpgUYXzJ1jVzUYsQAPYCRnfCTiC1WN2GnQXBUPtJoBo6n3no7ZCZAFqZB83ZCZCAc3r96sWYlNcykYkYQZDZD"
PAGE_ID = "508353805701957"
MESSAGE = f"""Training progress:
============================
Iteration: {1}
Loss: {1}
Loss_ce: {1}
    
"""

body = {
    "recipient": {
        "id": "9191294137603607"
    },
    "messaging_type": "RESPONSE",
    "message": {
        "text": MESSAGE
    }
}
url = f"https://graph.facebook.com/v21.0/{PAGE_ID}/messages?access_token={PAGE_ACCESS_TOKEN}"
requests.post(url, json=body, headers={"Content-Type": "application/json"})