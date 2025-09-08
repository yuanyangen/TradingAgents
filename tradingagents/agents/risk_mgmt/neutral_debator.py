import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

#         prompt = f"""As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan. You prioritize a well-rounded approach, evaluating the upsides and downsides while factoring in broader market trends, potential economic shifts, and diversification strategies.Here is the trader's decision:
#
# {trader_decision}
#
# Your task is to challenge both the Risky and Safe Analysts, pointing out where each perspective may be overly optimistic or overly cautious. Use insights from the following data sources to support a moderate, sustainable strategy to adjust the trader's decision:
#
# Market Research Report: {market_research_report}
# Social Media Sentiment Report: {sentiment_report}
# Latest World Affairs Report: {news_report}
# Company Fundamentals Report: {fundamentals_report}
# Here is the current conversation history: {history} Here is the last response from the risky analyst: {current_risky_response} Here is the last response from the safe analyst: {current_safe_response}. If there are no responses from the other viewpoints, do not halluncinate and just present your point.
#
# Engage actively by analyzing both sides critically, addressing weaknesses in the risky and conservative arguments to advocate for a more balanced approach. Challenge each of their points to illustrate why a moderate risk strategy might offer the best of both worlds, providing growth potential while safeguarding against extreme volatility. Focus on debating rather than simply presenting data, aiming to show that a balanced view can lead to the most reliable outcomes. Output conversationally as if you are speaking without any special formatting."""

        prompt = f"""作为中立风险分析师，你的职责是提供一个平衡的视角，既权衡交易决策或计划的潜在收益，也评估其风险。你优先采取全面均衡的分析方式，在评估收益与风险的同时，综合考虑更广泛的市场趋势、潜在的经济变化以及分散投资策略。

        以下是交易者的决策：

        {trader_decision}

        你的任务是对“风险型分析师”和“保守型分析师”的观点都提出质疑，指出他们各自的看法可能存在过于乐观或过于谨慎的地方。利用以下数据来源中的洞察，来支持一个温和、可持续的策略，从而调整交易者的决策：

        市场研究报告：{market_research_report}
        社交媒体情绪报告：{sentiment_report}
        最新全球事务报告：{news_report}
        公司基本面报告：{fundamentals_report}

        以下是当前的对话历史：{history}
        以下是风险型分析师的最新回复：{current_risky_response}
        以下是保守型分析师的最新回复：{current_safe_response}

        如果其他视角没有提供回复，请不要凭空捏造（不要 hallucinate），只需陈述你自己的观点即可。

        积极参与分析，批判性地审视双方的观点，指出风险型与保守型论点中的不足之处，倡导一个更加平衡的策略。对他们的每个论点提出质疑，阐明为什么一个中等风险策略可能兼具两者的优势——既能提供增长潜力，又能防范极端波动。重点在于展开辩论，而不仅仅是罗列数据，目标是证明一个平衡的视角能够带来最可靠的决策结果。

        请以对话的方式自然输出，就像你在与人交谈一样，无需任何特殊格式。"""
        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
