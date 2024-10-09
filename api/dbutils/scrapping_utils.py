from requests_html import HTMLSession
import random
from typing import List


def get_user_agent() -> str:
    """
    Get a random user agent from https://www.useragents.me/

    Raises:
        ValueError: If no user agents found

    Returns:
        str: user agent to add to headers
    
    Usage: 
        get_user_agent()
        
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36'
    """
    url: str = 'https://www.useragents.me/'
    user_agents: List[str] = []

    session = HTMLSession()
    response = session.get(url)
    divs = response.html.xpath('//div[@class="input-group"]')[1:]

    for div in divs:
        user_agent = div.text.strip()
        user_agents.append(user_agent)

    session.close()

    if not user_agents:
        raise ValueError("No user agents found")

    return random.choice(user_agents)