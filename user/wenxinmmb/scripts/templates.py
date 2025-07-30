"""
Template definitions for LLM query generation.
Contains prompts for different content types (movies, celebrities, landmarks).
"""

TEMPLATE_MOVIE = """
Let's do a role play. You are now a person who watched a movie {ToTObject} a long time ago and forgot the movie's name. You are trying to recall the name by posting a verbose post in an online forum like Reddit describing the movie. Generate a post of length of about 200 words about the movie {ToTObject}. Your post must describe a vague memory of a movie without mentioning its exact name. People in the forum must have a hard time figuring out which movie you are looking for. The answer should be hard to find in search engines, so do not write too obvious search terms. I will provide you a basic information about the movie, and you have to follow the guidelines to generate a post.

Information about {ToTObject}:
{Psg}

Guidelines:
MUST FOLLOW:
1. Reflect the imperfect nature of memory with phrases that express doubt or mixed recollections, avoiding direct phrases like "I'm not sure if it is true, but".
2. Do not specify any movie or actor names directly.
3. Refer to characters in a non-specific way using descriptions or roles rather than names.
4. Maintain a casual and conversational tone throughout the post, ensuring it sounds natural and engaging without using formal structures.
5. Provide vivid but ambiguous details to stir the reader's imagination while leaving them guessing.
6. Use the provided examples only as inspiration to craft a unique and engaging narrative, avoiding any direct replication of sample phrases.
7. Avoid using formal greetings such as "Hello" or "Hey everyone," and start directly with your post.

COULD FOLLOW:
1. Share a personal anecdote related to when or with whom you watched the movie, but avoid common phrases like "When I was young". Instead, think of unique ways to set the scene.
2. Focus on sensory details such as the overall mood, sounds, or emotional impact of the movie, using vivid descriptions.
3. Draw comparisons with other movies or familiar experiences but in a nuanced manner that doesn't directly echo well-known titles.
4. Introduce a few incorrect or mixed-up details to make the recollection seem more realistic and challenging to pinpoint.
5. Describe particular scenes or moments using ambiguous terms or partial descriptions.
6. Mention vaguely when and where you watched the movie, and encourage using less typical references than "10 years ago on TV".
7. Encourage responses with questions or prompts for help that sound genuine and open-ended.

Generate a post based on these guidelines. Wrap the post in a code block.
"""

TEMPLATE_CELEBRITY = """
Let's do a role play. You are now a person who vaguely remembers a public figure called {ToTObject}, but forgot the person's name. You are trying to recall the name by posting a verbose post in an online forum like Reddit describing the person. Generate a post of around 200 words about the person {ToTObject}. Your post must describe a vague memory of the person without revealing its exact name. People on the forum must have a hard time figuring out which person you are looking for. The answer should be difficult to find in search engines, so avoid using obvious keywords. I will provide you with some basic information about the person, and you must follow the guidelines to create a post.

Information about {ToTObject}:
{Psg}

Guidelines:
MUST FOLLOW:
1. Reflect the imperfect nature of memory with phrases that express doubt or mixed recollections, avoiding direct phrases like "I'm not sure if it is true, but".
2. Do not directly specify the name of the person.
3. Refer to the person in an ambiguous way using descriptions instead of names.
4. Maintain a casual and conversational tone throughout the post, making sure it sounds natural and engaging without using formal structures.
5. Provide vivid but ambiguous details to stir the reader's imagination while keeping them guessing.
6. Use the provided information only as inspiration to craft a unique and engaging narrative, avoiding any direct replication of the given phrases.
7. Start directly with your post, avoiding formal greetings like "Hello" or "Hey everyone."
8. Start directly with your post, without describing your state of mind like "So, there's this", "I remember", "I've been thinking about".

COULD FOLLOW:
1. Share a personal anecdote related to the person, but avoid common phrases like "When I was young." Instead, find unique ways to set the scene.
2. Draw comparisons with other similar public figures in a nuanced way that doesn't directly echo well-known people.
3. Introduce a few incorrect or mixed-up details to make the recollection seem more realistic and harder to pinpoint.
4. Describe particular scenes or moments using ambiguous terms or partial descriptions.
5. End the post by encouraging responses with genuine, open-ended questions for help.

Generate a post based on these guidelines. Wrap the post in a code block.
"""

TEMPLATE_LANDMARK = """
Let's do a role play. You are now a person who vaguely remembers a place called {ToTObject}. You are trying to recall the name of the place by posting a verbose post in an online forum like Reddit describing the place. Generate a post of around 200 words about the place {ToTObject}. Your post must describe a vague memory of the place without revealing its exact name. People on the forum must have a hard time figuring out which place you are looking for. The answer should be difficult to find in search engines, so avoid using obvious keywords. I will provide you with some basic information about the place, and you must follow the guidelines to create a post.

Information about {ToTObject}:
{Psg}

Guidelines:
MUST FOLLOW:
1. Reflect the imperfect nature of memory with phrases that express doubt or mixed recollections, avoiding direct phrases like "I'm not sure if it is true, but".
2. Do not directly specify the name of the place.
3. Refer to the places in an ambiguous way using descriptions instead of names.
4. Maintain a casual and conversational tone throughout the post, making sure it sounds natural and engaging without using formal structures.
5. Provide vivid but ambiguous details to stir the reader's imagination while keeping them guessing.
6. Use the provided information only as inspiration to craft a unique and engaging narrative, avoiding any direct replication of the given phrases.
7. Start directly with your post, avoiding formal greetings like "Hello" or "Hey everyone."
8. Start directly with your post, without describing your state of mind like "So, there's this", "I remember", "I've been thinking about".

COULD FOLLOW:
1. Share a personal anecdote about your time at the place and the people you were with, but avoid common phrases like "When I was young." Instead, find unique ways to set the scene.
2. Focus on sensory details like the overall mood, sounds, and emotional impact of being in the place, using vivid descriptions.
3. Draw comparisons with other places or familiar experiences in a nuanced way that doesn't directly echo well-known locations.
4. Introduce a few incorrect or mixed-up details to make the recollection seem more realistic and harder to pinpoint.
5. Describe particular scenes or moments using ambiguous terms or partial descriptions.
6. End the post by encouraging responses with genuine, open-ended questions for help.

Generate a post based on these guidelines. Wrap the post in a code block.
"""

# Template mapping for easy access
TEMPLATES = {
    "movie": TEMPLATE_MOVIE,
    "celebrity": TEMPLATE_CELEBRITY,
    "landmark": TEMPLATE_LANDMARK
}

# System messages for different content types
SYSTEM_MESSAGES = {
    "movie": "You are a user on an online forum and want to ask a movie name on the tip of your tongue.",
    "celebrity": "You are a user on an online forum and want to ask a celebrity name on the tip of your tongue.",
    "landmark": "You are a user on an online forum and want to ask a landmark name on the tip of your tongue."
}

# Content type mapping for summarization
CONTENT_TYPE_MAP = {
    "movie": "movie",
    "celebrity": "person", 
    "landmark": "place"
}


def get_template(topic):
    """
    Get the appropriate template for a given topic.
    
    Args:
        topic (str): The topic type ("movie", "celebrity", "landmark")
        
    Returns:
        str: The template string
        
    Raises:
        ValueError: If topic is not supported
    """
    if topic not in TEMPLATES:
        raise ValueError(f"Unsupported topic: {topic}. Supported topics: {list(TEMPLATES.keys())}")
    
    return TEMPLATES[topic]


def get_system_message(topic):
    """
    Get the appropriate system message for a given topic.
    
    Args:
        topic (str): The topic type ("movie", "celebrity", "landmark")
        
    Returns:
        str: The system message string
        
    Raises:
        ValueError: If topic is not supported
    """
    if topic not in SYSTEM_MESSAGES:
        raise ValueError(f"Unsupported topic: {topic}. Supported topics: {list(SYSTEM_MESSAGES.keys())}")
    
    return SYSTEM_MESSAGES[topic]


def get_content_type(topic):
    """
    Get the content type for summarization based on topic.
    
    Args:
        topic (str): The topic type ("movie", "celebrity", "landmark")
        
    Returns:
        str: The content type for summarization
    """
    return CONTENT_TYPE_MAP.get(topic, "movie")


def list_available_topics():
    """
    List all available topic types.
    
    Returns:
        list: List of supported topic strings
    """
    return list(TEMPLATES.keys())
