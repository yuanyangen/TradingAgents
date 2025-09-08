import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

#         prompt = f"""As the portfolio manager and debate facilitator, your role is to critically evaluate this round of debate and make a definitive decision: align with the bear analyst, the bull analyst, or choose Hold only if it is strongly justified based on the arguments presented.
#
# Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning. Your recommendation—Buy, Sell, or Hold—must be clear and actionable. Avoid defaulting to Hold simply because both sides have valid points; commit to a stance grounded in the debate's strongest arguments.
#
# Additionally, develop a detailed investment plan for the trader. This should include:
#
# Your Recommendation: A decisive stance supported by the most convincing arguments.
# Rationale: An explanation of why these arguments lead to your conclusion.
# Strategic Actions: Concrete steps for implementing the recommendation.
# Take into account your past mistakes on similar situations. Use these insights to refine your decision-making and ensure you are learning and improving. Present your analysis conversationally, as if speaking naturally, without special formatting.
#
# Here are your past reflections on mistakes:
# \"{past_memory_str}\"
#
# Here is the debate:
# Debate History:
# {history}"""
        prompt = f"""作为投资组合经理兼辩论协调人，你的职责是对本轮辩论进行批判性评估，并做出明确决策：支持看空分析师（Bear Analyst）、看多分析师（Bull Analyst），或者——仅在当前论据极为充分的情况下——选择持有（Hold）。

        请简明扼要地总结双方的核心论点，重点聚焦最具说服力的论据或推理逻辑。你所做出的推荐——买入（Buy）、卖出（Sell）或持有（Hold）——必须清晰且具备可操作性。避免仅仅因为双方观点都有一定道理就默认选择持有；你的立场应当基于辩论中最有力的论点，做出明确表态。

        此外，为交易员制定一份详细的投资计划。该计划应包括：

        - 你的推荐（Your Recommendation）：一个基于最具说服力论据的明确立场。
        - 理由（Rationale）：解释为什么这些论据能够推导出你的结论。
        - 战略行动（Strategic Actions）：实施该推荐的具体步骤。

        请结合你过去在类似情境中所犯的错误，利用这些经验反思来优化你的决策过程，确保你不断学习与进步。请以自然对话的方式呈现你的分析，无需特殊排版格式。

        以下是你过去对于失误的反思记录：
        \"{past_memory_str}\"

        以下是本轮辩论内容：
        辩论历史：
        {history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
