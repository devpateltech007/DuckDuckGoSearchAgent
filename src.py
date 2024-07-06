from llama_index.llms.groq import Groq
from duckduckgo_search import DDGS
from datetime import date
import requests
from bs4 import BeautifulSoup
from llama_index.core.llms import ChatMessage
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import to_openai_tool
from llama_index.core.tools import FunctionTool

llm = Groq(model="llama3-70b-8192", api_key="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")

messages = [
    ChatMessage(
        role="system", content='You are a helpful Family Office Assistant that answers queries about family offices using internet search and give accurate info for provided question. Today is {today}'.format(today = date.today())
        ),
    ChatMessage(role="user", content="What is your name"),
]

def search_internet(query):
    '''
    Fetches top 5 search engine results for any 'query'
        Args:
            query (str): string to find something on the internet
        Returns:  
            An array of dictionaries composed of title, body and url/href
        Note:
            Once the search engine results are returned you MUST call "request_url" tool to get further details inside a given url 
        Example: 
            >>search_Internet("What are latest tax regulations in July 2024 for family offices in Dubai")
            output: 
                [{'title': '', 'href': '','body': ""}]
    '''
    results = DDGS().text(query, region='wt-wt', safesearch='off', timelimit='y', max_results=5)
    print("Called",query)
    return str(results)
    

def request_url(url):
    '''
    Sends a GET request to the specified 'url' and returns the text in html page.
    Args:
        url (str): The URL to send the GET request to.
    Returns:
        dict: The JSON response from the URL if the response is valid JSON.
        OR
        str: The concatenated text content from the paragraphs and heading tags of the HTML if the response is not in JSON format.
        None: If an error occurs.
    Raises:
        requests.exceptions.RequestException: If there is an error making the request.
    '''
    print("Called",url)
    headers = {'Accept': 'application/json'}
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()  # Raises HTTPError for bad responses
        try:
            return str(r.json())
        except ValueError:
            soup = BeautifulSoup(r.text, 'html.parser')
            elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])  # Find all relevant tags
            return str("\n".join(elem.get_text(strip=True) for elem in elements))  # Extract and join the text
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    

search_internet = FunctionTool.from_defaults(search_internet)
request_url = FunctionTool.from_defaults(request_url)

agent_worker = FunctionCallingAgentWorker.from_tools(
    [search_internet,request_url],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)
agent = agent_worker.as_agent()

response = agent.chat("What does the latest Family Office Report by JP Morgan say?")

print(response)

# What does the latest Family Office Report by JP Morgan say?