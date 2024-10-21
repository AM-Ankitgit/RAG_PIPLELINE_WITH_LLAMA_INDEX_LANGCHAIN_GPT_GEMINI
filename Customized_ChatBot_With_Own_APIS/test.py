import os
import requests

# Set your Google Gemini API Key
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# Sample Llama Index - A dictionary to map intents to actions
llama_index = {
    "milk_entry": {
        "keywords": ["enter milk", "milk entery", "milk entry", "add milk"],
        "api_endpoint": "/api/milk-entry"
    },
    "milk_history": {
        "keywords": ["milk history", "show milk history", "milk histiry"],
        "api_endpoint": "/api/milk-history"
    }
}

def get_gemini_intent(user_input):
    """
    Function to get intent using Google Gemini API
    """
    url = "https://api.google.com/gemini/v1/intent"
    headers = {
        "Authorization": f"Bearer {GOOGLE_GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "text": user_input
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get("intent")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def match_intent_using_llama_index(user_input):
    """
    Match user input to predefined intents using Llama Index
    """
    for intent, details in llama_index.items():
        for keyword in details["keywords"]:
            if keyword in user_input.lower():
                return intent
    return None

def handle_intent(intent, user_input):
    """
    Function to handle specific intents by calling relevant API
    """
    if intent in llama_index:
        api_endpoint = llama_index[intent]["api_endpoint"]
        # Here we are simulating API interaction
        print(f"Calling API at {api_endpoint} for user input: '{user_input}'")
        # Simulate API call and response
        response = requests.post(f"http://your-server.com{api_endpoint}", json={"input": user_input})
        return response.json()

    return "Sorry, I didn't understand that."

# Main chatbot loop
def chatbot():
    print("Chatbot is running... Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        # First try to match intent using Llama Index
        intent = match_intent_using_llama_index(user_input)
        
        # If no intent is found, try using Google Gemini
        if not intent:
            intent = get_gemini_intent(user_input)

        # Handle the recognized intent
        response = handle_intent(intent, user_input)
        print(f"Chatbot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
