import random
import time

def current_weather_prompt_generator() -> str:
    """Default prompt generator"""
    cities = [
        "Chicago (41.9,-87.6)", "New York (40.7,-74.0)", "Los Angeles (34.1,-118.2)", "Houston (29.8,-95.4)",
        "Phoenix (33.4,-112.1)", "Philadelphia (39.9,-75.2)", "San Antonio (29.4,-98.5)", "San Diego (32.7,-117.2)",
    ]
    prompts = [
        f"What is the weather in {random.choice(cities)} today?",
    ]
    return random.choice(prompts)

def weather_forecast_prompt_generator() -> str:
    """Custom prompt generator for weather forecast"""
    cities = [
        "Chicago (41.9,-87.6)", "New York (40.7,-74.0)", "Los Angeles (34.1,-118.2)", "Houston (29.8,-95.4)",
        "Phoenix (33.4,-112.1)", "Philadelphia (39.9,-75.2)", "San Antonio (29.4,-98.5)", "San Diego (32.7,-117.2)",
    ]
    prompts = [
        f"Today is {time.strftime('%Y-%m-%d')}. What is the weather forecast for {random.choice(cities)} tomorrow?",
    ]
    return random.choice(prompts)

def web_search_prompt_generator() -> str:
    """Custom prompt generator for web search tools"""
    topics = [
        "latest technology trends",
        "current events in politics",
        "recent advancements in AI",
        "top travel destinations in 2026",
        "health benefits of meditation",
    ]
    prompts = [
        f"Can you provide information about {random.choice(topics)} using web search?",
    ]
    return random.choice(prompts)

def paper_search_prompt_generator() -> str:
    """Prompt generator for academic paper search tools"""
    topics = [
        "transformer architecture in natural language processing",
        "reinforcement learning from human feedback",
        "large language model alignment",
        "diffusion models for image generation",
        "graph neural networks",
        "federated learning and privacy",
        "neural architecture search",
        "multi-modal learning",
        "retrieval-augmented generation",
        "model compression and knowledge distillation",
        "chain-of-thought prompting",
        "vision-language models",
        "in-context learning",
        "model merging and weight averaging",
        "AI safety and interpretability",
    ]
    topic = random.choice(topics)
    prompts = [
        f"Search for recent academic papers about {topic}.",
        f"Find research papers on {topic}.",
        f"Look up the latest papers related to {topic}.",
    ]
    return random.choice(prompts)

def us_stock_price_prompt_generator() -> str:
    """Prompt generator for US stock current price query tools"""
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "TSLA", "META", "NFLX", "JPM", "V",
        "BRK.B", "JNJ", "WMT", "PG", "MA",
        "UNH", "HD", "BAC", "XOM", "CVX",
    ]
    ticker = random.choice(tickers)
    prompts = [
        f"What is the current stock price of {ticker}?",
        f"Get the latest price for {ticker} stock.",
        f"What is {ticker} trading at right now?",
        f"How much is {ticker} worth per share today?",
    ]
    return random.choice(prompts)
