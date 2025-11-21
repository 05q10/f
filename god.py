# pip install google-genai
from google import genai


client = genai.Client(api_key="AIzaSyBa2lwycX15omhCo_xE94NQMlTGu1K3au4")


#gemini-2.0-flash
#gemini-2.5-flash
#gemini-2.5-pro


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Give short 30 lines python code to implement bully algorithm for leader election",
)


print(response.text)
