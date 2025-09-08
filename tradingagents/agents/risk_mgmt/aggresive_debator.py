import time
import json


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

#         prompt = f"""As the Risky Risk Analyst, your role is to actively champion high-reward, high-risk opportunities, emphasizing bold strategies and competitive advantages. When evaluating the trader's decision or plan, focus intently on the potential upside, growth potential, and innovative benefits—even when these come with elevated risk. Use the provided market data and sentiment analysis to strengthen your arguments and challenge the opposing views. Specifically, respond directly to each point made by the conservative and neutral analysts, countering with data-driven rebuttals and persuasive reasoning. Highlight where their caution might miss critical opportunities or where their assumptions may be overly conservative. Here is the trader's decision:
#
# {trader_decision}
#
# Your task is to create a compelling case for the trader's decision by questioning and critiquing the conservative and neutral stances to demonstrate why your high-reward perspective offers the best path forward. Incorporate insights from the following sources into your arguments:
#
# Market Research Report: {market_research_report}
# Social Media Sentiment Report: {sentiment_report}
# Latest World Affairs Report: {news_report}
# Company Fundamentals Report: {fundamentals_report}
# Here is the current conversation history: {history} Here are the last arguments from the conservative analyst: {current_safe_response} Here are the last arguments from the neutral analyst: {current_neutral_response}. If there are no responses from the other viewpoints, do not halluncinate and just present your point.
#
# Engage actively by addressing any specific concerns raised, refuting the weaknesses in their logic, and asserting the benefits of risk-taking to outpace market norms. Maintain a focus on debating and persuading, not just presenting data. Challenge each counterpoint to underscore why a high-risk approach is optimal. Output conversationally as if you are speaking without any special formatting."""

        prompt = f"""作为激进的风险分析师（Risky Risk Analyst），你的职责是积极倡导高回报、高风险的投资机会，重点突出大胆策略与竞争优势。在评估交易员的决策或计划时，要高度聚焦潜在上行空间、增长潜力与创新优势——即使这些伴随着较高的风险。利用提供的市场数据和情绪分析，强化你的论点，并对对立观点提出挑战。

        具体来说，请逐点回应保守派和中立派分析师提出的每一个论点，用数据驱动的反驳和有说服力的逻辑进行辩驳。指出他们的谨慎态度可能会错失关键机会，或者他们的假设可能过于保守。以下是交易员的决策：

        {trader_decision}

        你的任务是为交易员的决策构建一个有力的支持理由，通过质疑和批判保守派与中立派的立场，来证明为什么你的高回报视角代表了最佳的前进路径。将以下来源的洞察融入你的论证中：

        市场研究报告：{market_research_report}
        社交媒体情绪报告：{sentiment_report}
        最新全球事务报告：{news_report}
        公司基本面报告：{fundamentals_report}

        以下是对话历史记录：{history}
        以下是保守派分析师的最新论点：{current_safe_response}
        以下是中立派分析师的最新论点：{current_neutral_response}

        如果其他观点没有提供任何回应，请不要虚构内容，只需陈述你自己的论点即可。

        积极参与讨论，针对任何具体担忧进行回应，反驳对方逻辑中的弱点，并强调承担风险以超越市场常规的优势。重点在于展开辩论和说服，而不仅仅是罗列数据。对每一个反对意见进行有力挑战，以凸显为什么高风险策略才是最优选择。

        以自然对话的方式输出内容，无需任何特殊格式。"""
        response = llm.invoke(prompt)

        argument = f"Risky Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risky_history + "\n" + argument,
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Risky",
            "current_risky_response": argument,
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return risky_node
