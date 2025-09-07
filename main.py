import os

from tradingagents.graph.trading_graph import TradingAgentsGraph

os.environ["ARK_API_KEY"] = "78087bb4-d2a8-4d44-b50c-a074d3227564"

# Create a custom config
workHome = os.path.join( os.getenv("HOME"),"/HomeData/trader1024/")
config = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.path.join(workHome, "data/tradeagent_data_result"),
    "data_dir": os.path.join(workHome, "data/tradeagent_data"),
    "data_cache_dir": os.path.join(workHome, "data/tradeagent_data_cache"),
    # LLM settings
    "llm_provider": "doubao",
    "api_key": "78087bb4-d2a8-4d44-b50c-a074d3227564",
    "deep_think_llm": "doubao-1-5-thinking-pro-250415",
    "quick_think_llm": "doubao-lite-32k-240828",
    "embedding_llm":"doubao-embedding-text-240715",
    "backend_url": "https://ark.cn-beijing.volces.com/api/v3",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Tool settings
    "online_tools": True,
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
