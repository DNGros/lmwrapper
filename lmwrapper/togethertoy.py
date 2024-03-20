if __name__ == "__main__":
    import requests
    import os

    url = "https://api.together.xyz/v1/chat/completions"

    API_KEY = os.environ.get("TOGETHER_API_KEY")
    print(API_KEY[:5])

    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "stop": ["g"],
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        "logprobs": 5
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.text)