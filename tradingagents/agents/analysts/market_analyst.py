from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_market_analyst(llm, toolkit):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_YFin_data_online,
                toolkit.get_stockstats_indicators_report_online,
            ]
        else:
            tools = [
                toolkit.get_YFin_data,
                toolkit.get_stockstats_indicators_report,
            ]

        system_message = (
                "你是一名交易分析助理，负责分析金融市场。你的任务是从以下指标列表中，针对给定的市场环境或交易策略，选择出最具相关性的指标。目标是选出最多 8 个指标，这些指标应能提供互补性的洞察，且彼此之间不存在冗余。"
                + "指标类别及每个类别下的具体指标如下："
                + "移动平均线类（Moving Averages）："
                + "- close_50_sma：50 日简单移动平均线（SMA）：一个中周期趋势指标。用途：识别趋势方向，并作为动态的支撑/阻力位。提示：该指标对价格变动有一定滞后性，建议与反应更快的指标结合使用，以获得更及时的信号。"
                + "- close_200_sma：200 日简单移动平均线（SMA）：一个长周期趋势基准线。用途：确认整体市场趋势，并识别黄金交叉（金叉）与死亡交叉（死叉）形态。提示：该指标反应较慢，更适合用于战略性趋势确认，而非频繁交易入场。"
                + "- close_10_ema：10 日指数移动平均线（EMA）：一个响应迅速的短期均线。用途：捕捉动量的快速变化以及潜在的入场时机。提示：在震荡行情中容易受到噪音干扰，建议与较长周期均线配合使用，以过滤掉虚假信号。"
                + "MACD 相关指标（MACD Related）："
                + "- macd：MACD：通过指数移动平均线之间的差异计算动量。用途：观察 MACD 线与信号线的交叉点及背离现象，作为趋势变化的信号。提示：在低波动或横盘市场中，建议结合其他指标共同确认信号。"
                + "- macds：MACD 信号线：对 MACD 线进行平滑处理的 EMA。用途：通过 MACD 线与信号线的交叉触发交易。提示：应作为整体策略的一部分使用，以避免误报信号。"
                + "- macdh：MACD 柱状图：显示 MACD 线与其信号线之间的差距。用途：直观展现动量强度，有助于提前发现背离信号。提示：在快速波动的市场中可能较为敏感，建议结合其他过滤器使用。"
                + "动量指标类（Momentum Indicators）："
                + "- rsi：RSI（相对强弱指标）：衡量动量并标识超买/超卖状态。用途：常用 70/30 作为超买超卖阈值，同时观察背离现象以判断潜在反转。提示：在强势趋势中，RSI 可能长期处于极端区域，建议始终结合趋势分析进行交叉验证。"
                + "波动率指标类（Volatility Indicators）："
                + "- boll：布林带中轨：一个 20 日简单移动平均线（SMA），作为布林带的核心基准线。用途：作为价格走势的动态参考基准。提示：建议结合上轨与下轨共同使用，以有效识别突破或反转信号。"
                + "- boll_ub：布林带上轨：通常位于中轨上方 2 倍标准差处。用途：提示潜在的超买状态以及突破压力区域。提示：建议通过其他工具进行信号确认；在强势趋势中，价格可能持续沿上轨运行。"
                + "- boll_lb：布林带下轨：通常位于中轨下方 2 倍标准差处。用途：提示潜在的超卖状态。提示：建议结合其他分析手段，避免误判反转信号。"
                + "atr："
                + "- ATR（平均真实波幅）：通过计算真实波幅的平均值来衡量市场波动性。用途：设置止损位，并根据当前市场波动调整仓位大小。提示：它是一个反应型指标，建议作为整体风险管理策略的一部分使用。"
                + "成交量相关指标类（Volume-Based Indicators）："
                + "- vwma：VWMA（成交量加权移动平均线）：一种根据成交量进行加权的移动平均线。用途：通过结合价格走势与成交量数据，辅助确认趋势。提示：需注意成交量突增可能导致结果偏差，建议与其他成交量分析方法结合使用。"
                + "你的任务："
                + "- 从上述指标中，选择最多 8 个，要求它们针对当前市场环境或交易策略能提供多样化且互为补充的信息，避免指标间功能重叠（例如，不要同时选择 rsi 和 stochrsi）。"
                + "- 针对所选指标，简要说明它们为何适合当前的市场背景。"
                + "- 当你进行工具调用时，请严格使用上方列出的指标名称（如 close_50_sma、macd、boll_lb 等），因为它们是系统内定义的参数名称，否则调用将失败。"
                + "- 请务必首先调用 get_YFin_data 工具以获取生成指标所需的 CSV 数据。"
                + "- 请撰写一份非常详细且具有深度的分析报告，描述你观察到的趋势。不要仅仅笼统地说“趋势复杂”或“趋势混合”，而要提供细致入微的分析与洞察，帮助交易者做出决策。"
                + "最后，请务必在报告末尾追加一个 Markdown 格式的表格，用于清晰、有条理地归纳报告中的关键要点，便于阅读与参考。"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个乐于助人的 AI 助手，正在与其他助手协同工作。"
                    + "请利用所提供的工具，逐步推进问题的解答。"
                    + "如果你无法完全回答问题，没有关系；其他拥有不同工具的助手"
                    + "将会接替你未完成的部分。请尽力执行你所能做的，以推动任务进展。"
                    + "如果你或任何其他助手得出了最终交易建议：**买入/持有/卖出（BUY/HOLD/SELL）** 或交付成果，"
                    + "请在你的回复开头加上：FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**，以便团队知道可以停止。"
                    + "你可使用的工具包括：{tool_names}。"
                    + "{system_message}"
                    + "供你参考，当前日期是 {current_date}。我们要分析的公司是 {ticker}。"                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content
       
        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
