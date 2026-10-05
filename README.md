<p align="center">
  <img src="assets/TauricResearch.png" style="width: 60%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2412.20138" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.20138-B31B1B?logo=arxiv"/></a>
  <a href="https://discord.com/invite/hk9PGKShPK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-TradingResearch-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="https://x.com/TauricResearch" target="_blank"><img alt="X Follow" src="https://img.shields.io/badge/X-TauricResearch-white?logo=x&logoColor=white"/></a>
  <a href="https://github.com/TauricResearch/" target="_blank"><img alt="Community" src="https://img.shields.io/badge/GitHub_Community-TauricResearch-14C290?logo=discourse"/></a>
</div>
<br>
<div align="center">
  <a href="https://github.com/TauricResearch" target="_blank"><img alt="TradingAgents #1 Repository of the Day" src="https://trendshift.io/api/badge/repositories/16192" width="250" height="55"/></a>
</div>
<br>
<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=de">Deutsch</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=es">Español</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=fr">français</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ja">日本語</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ko">한국어</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=pt">Português</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ru">Русский</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=zh">中文</a>
</div>

---

# TradingAgents: Multi-Agents LLM Financial Trading Framework

## News

<!-- news:start -->
- [2026-09] **TradingAgents v0.5.2** released with parallel analysts for a faster analysis, a CLI that runs without prompts from flags such as `--ticker` and `--date`, the run's settings recorded in every report, and backtests that see only data published by each analysis date.
- [2026-09] **TradingAgents v0.5.1** released with a package layout organised by what each module holds (import paths moved), optional Jev screening of social posts, GPT-6 Sol and Luna as the default models, and fixes to run isolation and SEC EDGAR statements.
- [2026-09] **TradingAgents v0.5.0** released with point-in-time integrity across every dated path, SEC EDGAR fundamentals served as filed, backtesting over a ticker and date grid, portfolio-aware runs, and current model lineups across every provider.

Full release notes are in [CHANGELOG.md](CHANGELOG.md).

<details>
<summary>Earlier news</summary>

- [2026-08] **TradingAgents v0.4.0** released with look-ahead / point-in-time fixes across FRED macro, social sentiment, and the decision-log memory; clearer decision signals; working CLI checkpoint resume; Trader price grounding; and the GPT-5.6 and GLM-5.3 models.
- [2026-07] **TradingAgents v0.3.1** released with correctness and stability fixes: Alpha Vantage look-ahead filtering, graph-router crash-safety, graph-shape-aware checkpoint resume, working crypto sentiment sources, a configurable LLM retry budget, Bedrock API-key auth, and Claude Sonnet 5 / Fable 5 support.
- [2026-06] **TradingAgents v0.3.0** released with a verified data-access contract, an expanded provider registry (NVIDIA, Kimi, Groq, Mistral, Bedrock, and any OpenAI-compatible endpoint), FRED and Polymarket data vendors, a current-generation model catalog, and a CI gate.
- [2026-05] **TradingAgents v0.2.5** released with the grounded Sentiment Analyst, GPT-5.5 etc. model coverage, Qwen/GLM/MiniMax dual-region support, `TRADINGAGENTS_*` env-var configurability with API-key auto-detection, remote Ollama support, non-US alpha benchmarks, and ticker path-traversal hardening.
- [2026-04] **TradingAgents v0.2.4** released with structured-output agents (Research Manager, Trader, Portfolio Manager), LangGraph checkpoint resume, persistent decision log, DeepSeek/Qwen/GLM/Azure provider support, Docker, and a Windows UTF-8 encoding fix.
- [2026-03] **TradingAgents v0.2.3** released with multi-language support, GPT-5.4 family models, unified model catalog, backtesting date fidelity, and proxy support.
- [2026-03] **TradingAgents v0.2.2** released with GPT-5.4/Gemini 3.1/Claude 4.6 model coverage, five-tier rating scale, OpenAI Responses API, Anthropic effort control, and cross-platform stability.
- [2026-02] **TradingAgents v0.2.0** released with multi-provider LLM support (GPT-5.x, Gemini 3.x, Claude 4.x, Grok 4.x) and improved system architecture.
- [2026-01] **Trading-R1** [Technical Report](https://arxiv.org/abs/2509.11420) released, with [Terminal](https://github.com/TauricResearch/Trading-R1) expected to land soon.

</details>
<!-- news:end -->

<div align="center">

🚀 [TradingAgents](#tradingagents-framework) | ⚡ [Installation & CLI](#installation-and-cli) | 🎬 [Demo](https://www.youtube.com/watch?v=90gr5lwjIho) | 📦 [Package Usage](#tradingagents-package) | 🤝 [Contributing](#contributing) | 📄 [Citation](#citation)

</div>

> 🎉 **TradingAgents** officially released! We have received numerous inquiries about the work, and we would like to express our thanks for the enthusiasm in our community.
>
> So we decided to fully open-source the framework. Looking forward to building impactful projects with you!

## TradingAgents Framework

TradingAgents is a multi-agent trading framework that mirrors the dynamics of real-world trading firms. By deploying specialized LLM-powered agents: from fundamental analysts, sentiment experts, and technical analysts, to trader, risk management team, the platform collaboratively evaluates market conditions and informs trading decisions. Moreover, these agents engage in dynamic discussions to pinpoint the optimal strategy.

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> TradingAgents framework is designed for research purposes. Trading performance may vary based on many factors, including the chosen backbone language models, model temperature, trading periods, the quality of data, and other non-deterministic factors. [It is not intended as financial, investment, or trading advice.](https://tauric.ai/disclaimer/)

Our framework decomposes complex trading tasks into specialized roles.

### Analyst Team
- Fundamentals Analyst: Evaluates company financials and performance metrics, identifying intrinsic values and potential red flags.
- Sentiment Analyst: Aggregates news headlines, StockTwits, and Reddit chatter into a single sentiment read to gauge short-term market mood.
- News Analyst: Monitors global news and macroeconomic indicators, interpreting the impact of events on market conditions.
- Technical Analyst: Utilizes technical indicators (like MACD and RSI) to detect trading patterns and forecast price movements.

The selected analysts work at the same time, each on its own tools, and the research debate starts once all of their reports are in.

<p align="center">
  <img src="assets/analyst.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### Researcher Team
- Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team. Through structured debates, they balance potential gains against inherent risks.

<p align="center">
  <img src="assets/researcher.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Trader Agent
- Composes reports from the analysts and researchers to make informed trading decisions, determining the timing and magnitude of trades.

<p align="center">
  <img src="assets/trader.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Risk Management and Portfolio Manager
- Continuously evaluates portfolio risk by assessing market volatility, liquidity, and other risk factors. The risk management team evaluates and adjusts trading strategies, providing assessment reports to the Portfolio Manager for final decision.
- The Portfolio Manager approves/rejects the transaction proposal. If approved, the order will be sent to the simulated exchange and executed.

<p align="center">
  <img src="assets/risk.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

## Installation and CLI

### Installation

Clone TradingAgents:
```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
```

TradingAgents needs Python 3.11 or later. Create a virtual environment in any of your favorite environment managers:
```bash
conda create -n tradingagents python=3.13
conda activate tradingagents
```

Or with [uv](https://docs.astral.sh/uv/):
```bash
uv venv --python 3.13
source .venv/bin/activate
```

Install the package and its dependencies (`uv pip install .` with uv):
```bash
pip install .
```

### Docker

Alternatively, run with Docker:
```bash
cp .env.example .env  # add your API keys
docker compose run --rm tradingagents
```

After updating the repository, rebuild the image with `docker compose build`.

Results, reports, the memory log and the cache live in the `tradingagents_data` volume. To keep them in a folder on the host instead, create the folder and point `TRADINGAGENTS_DATA_DIR` at it, in `.env` or the shell: `mkdir -p data && TRADINGAGENTS_DATA_DIR=./data docker compose run --rm tradingagents`.

For local models with Ollama:
```bash
docker compose --profile ollama run --rm tradingagents-ollama
```

### Required APIs

TradingAgents supports multiple LLM providers. Set the API key for your chosen provider:

```bash
export OPENAI_API_KEY=...          # OpenAI (GPT)
export GOOGLE_API_KEY=...          # Google (Gemini)
export ANTHROPIC_API_KEY=...       # Anthropic (Claude)
export XAI_API_KEY=...             # xAI (Grok)
export DEEPSEEK_API_KEY=...        # DeepSeek
export DASHSCOPE_API_KEY=...       # Qwen (international, dashscope-intl.aliyuncs.com)
export DASHSCOPE_CN_API_KEY=...    # Qwen (China, dashscope.aliyuncs.com)
export ZHIPU_API_KEY=...           # GLM via Z.AI (international)
export ZHIPU_CN_API_KEY=...        # GLM via BigModel (China, open.bigmodel.cn)
export MINIMAX_API_KEY=...         # MiniMax (global, api.minimax.io)
export MINIMAX_CN_API_KEY=...      # MiniMax (China, api.minimaxi.com)
export OPENROUTER_API_KEY=...      # OpenRouter
export MISTRAL_API_KEY=...         # Mistral
export MOONSHOT_API_KEY=...        # Kimi (Moonshot)
export GROQ_API_KEY=...            # Groq
export NVIDIA_API_KEY=...          # NVIDIA NIM
export FRED_API_KEY=...            # FRED macro data (free, optional)
export ALPHA_VANTAGE_API_KEY=...   # Alpha Vantage
export TYPESAFE_API_KEY=...        # Jev social-post screening (optional)
```

For Azure OpenAI, copy `.env.enterprise.example` to `.env.enterprise` and fill in your credentials.

For AWS Bedrock, install the extra with `pip install ".[bedrock]"`, set `llm_provider: "bedrock"`, configure AWS credentials (environment variables, `~/.aws/credentials`, or an IAM role) and `AWS_DEFAULT_REGION`, and use a Bedrock model ID, e.g. `us.anthropic.claude-opus-5-5`.

For local models, configure Ollama with `llm_provider: "ollama"`. The default endpoint is `http://localhost:11434/v1`; set `OLLAMA_BASE_URL` to point at a remote `ollama-serve`. Pull models with `ollama pull <name>`, and pick "Custom model ID" in the CLI for any model not listed by default.

For any other OpenAI-compatible server (vLLM, LM Studio, llama.cpp, or a custom relay), use `llm_provider: "openai_compatible"` and set the endpoint via `backend_url` (or `TRADINGAGENTS_LLM_BACKEND_URL`), e.g. `http://localhost:8000/v1` for vLLM or `http://localhost:1234/v1` for LM Studio. The model is whatever your server serves. No key is needed for local servers; set `OPENAI_COMPATIBLE_API_KEY` when the endpoint requires one.

With `TYPESAFE_API_KEY` set, the Sentiment Analyst screens StockTwits and Reddit posts with TypeSafe's Jev before reading them. Posts that are not about the company are dropped, and each source opens with a count of the remaining posts by stance: bullish, bearish, neutral, or unclear. Without the key, posts pass through unscreened. `jev-latest` moves with new releases; set `TYPESAFE_DEFAULT_MODEL` to a versioned ID such as `jev-1.13.0` to hold it fixed across runs.

Alternatively, copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### CLI Usage

Launch the interactive CLI:
```bash
tradingagents          # installed command
python -m cli.main     # alternative: run directly from source
```
You will see a screen where you can select your desired tickers, analysis date, LLM provider, research depth, and more. Your previous run's answers come back as the defaults, so pressing Enter accepts them. The `TRADINGAGENTS_*` variables in `.env` still skip their step entirely.

To run without questions, for a scheduled job or a script, answer the per-run steps with flags and the rest with `TRADINGAGENTS_*` variables:
```bash
export TRADINGAGENTS_LLM_PROVIDER=openai TRADINGAGENTS_QUICK_THINK_LLM=gpt-6-luna TRADINGAGENTS_DEEP_THINK_LLM=gpt-6-sol
export TRADINGAGENTS_OUTPUT_LANGUAGE=English TRADINGAGENTS_MAX_DEBATE_ROUNDS=1 TRADINGAGENTS_MAX_RISK_ROUNDS=1
tradingagents --ticker NVDA --date 2026-09-23 --analysts market,news,fundamentals --save --no-show
```
Each flag skips only its own question. Run without a terminal, a missing answer stops the run before it starts and names the flag or variable to set.

### Markets and tickers

TradingAgents works with any market Yahoo Finance covers, using the exchange-suffixed ticker. Company identity and the alpha benchmark resolve automatically per market.

- US: `AAPL`, `SPY`
- Hong Kong: `0700.HK` · Tokyo: `7203.T` · London: `AZN.L`
- India: `RELIANCE.NS`, `.BO` · Canada: `.TO` · Australia: `.AX`
- China A-shares: Shanghai `.SS`, Shenzhen `.SZ` (e.g. `600519.SS` for Kweichow Moutai)
- Crypto: `BTC-USD`, `ETH-USD`

<p align="center">
  <img src="assets/cli/cli_init.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

An interface will appear showing results as they load, letting you track the agent's progress as it runs.

<p align="center">
  <img src="assets/cli/cli_news.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

<p align="center">
  <img src="assets/cli/cli_transaction.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

## TradingAgents Package

### Implementation Details

We built TradingAgents with LangGraph to ensure flexibility and modularity. The framework supports multiple LLM providers: OpenAI, Google, Anthropic, xAI, DeepSeek, Qwen (Alibaba DashScope, international and China endpoints), GLM (Zhipu), MiniMax (global + China), OpenRouter, Ollama for local models, and Azure OpenAI for enterprise.

### Python Usage

To use TradingAgents inside your code, you can import the `tradingagents` module and initialize a `TradingAgentsGraph()` object. The `.propagate()` function will return a decision. You can run `main.py`, here's also a quick example:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())

# forward propagate
state, decision = ta.propagate("NVDA", "2026-09-01")
print(decision)

# the same report tree the CLI saves, under results_dir/reports
ta.save_reports(state, "NVDA")
```

You can also adjust the default configuration to set your own choice of LLMs, debate rounds, etc.

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"        # e.g. openai, google, anthropic, deepseek, groq, ollama; openai_compatible covers any OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp, ...)
config["deep_think_llm"] = "gpt-6-sol"    # Model for complex reasoning
config["quick_think_llm"] = "gpt-6-luna"   # Model for quick tasks
config["max_debate_rounds"] = 2

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2026-09-01")
print(decision)
```

See `tradingagents/default_config.py` for all configuration options.

### Fundamentals as filed

US company statements come from SEC EDGAR, which records the date every figure was filed. A run dated in the past reads the statements exactly as they stood that day: a fiscal year that has ended but has not been filed yet is not served, and a figure restated later still reads as first reported. Apple's 2008 total assets were filed as $39.6B and restated to $36.2B in 2010, so a run dated in between reads $39.6B. EDGAR needs no account or API key.

Other companies' statements come from Yahoo Finance, which dates a statement by the period it covers rather than by when it was published. A run dated today reads them; a run dated in the past is told they are withheld, since Yahoo cannot say which figures were public by then.

Insider trades are dated by when they happened, not when they were filed, so a run dated in the past is told they are withheld as well.

SEC asks callers to identify themselves and refuses requests that carry no contact address, so a default one is sent. Set your own so SEC can reach you rather than the project:

```bash
SEC_EDGAR_USER_AGENT="Your Name your@email.com"
```

It covers companies that file with the SEC, including foreign companies listed in the US. Anything else, such as Hong Kong or A-share listings, falls through to the next vendor in the chain. EDGAR's machine-readable filings begin in 2009, and a fourth quarter is reported as unavailable rather than derived, because filers publish it only inside the annual figure.

### Current holdings

By default the agents do not know what you hold, so their guidance is written for a reader who applies it to their own position. Pass a portfolio to have the trader, the risk analysts and the portfolio manager work against your actual book.

```python
from tradingagents.portfolio import PortfolioContext

portfolio = PortfolioContext.model_validate({
    "cash": 25000.0,
    "currency": "USD",
    "positions": [{"ticker": "NVDA", "quantity": 120, "average_price": 150.0}],
})
_, decision = ta.propagate("NVDA", "2026-09-01", portfolio=portfolio)
```

The CLI takes the same content as a JSON file: `tradingagents --portfolio my_book.json`.

An empty `positions` list means a flat book, which is different from passing nothing. A run without a portfolio is never treated as flat.

## Persistence and Recovery

TradingAgents persists two kinds of state across runs.

### Memory log

The memory log is always on. Each completed run appends its decision to `~/.tradingagents/memory/trading_memory.md`. On the next run for the same ticker, TradingAgents fetches the realised return (raw, and alpha against the instrument's regional benchmark), generates a one-paragraph reflection, and injects the most recent same-ticker decisions plus recent cross-ticker lessons into the Portfolio Manager prompt, so each analysis carries forward what worked and what didn't.

Override the path with `TRADINGAGENTS_MEMORY_LOG_PATH`.

### Checkpoint resume

Checkpoint resume is opt-in via `--checkpoint`. When enabled, LangGraph saves state after each node so a crashed or interrupted run resumes from the last successful step instead of starting over. The run view says whether it resumed a saved run or started fresh. Checkpoints are cleared automatically on successful completion.

Per-ticker SQLite databases live at `~/.tradingagents/cache/checkpoints/<TICKER>.db` (override the base with `TRADINGAGENTS_CACHE_DIR`). Use `--clear-checkpoints` to reset all of them before a run.

```bash
tradingagents --checkpoint           # enable for this run
tradingagents --clear-checkpoints    # reset before running
```

```python
config = DEFAULT_CONFIG.copy()
config["checkpoint_enabled"] = True
ta = TradingAgentsGraph(config=config)
_, decision = ta.propagate("NVDA", "2026-09-01")
```

## Evaluating decisions over time

One run gives one decision, which cannot tell you whether the system decides well. `run_backtest` runs the same pipeline over a grid of tickers and dates, writes to a memory log of its own, and scores the decisions whose holding window has since traded.

```python
from tradingagents.backtest import iter_grid, run_backtest, summarize

dates = iter_grid("2026-06-01", "2026-08-01", every_n_days=7)
result = run_backtest(["NVDA", "AAPL"], dates, config, selected_analysts=["market", "news"])
print(summarize(result).render())
```

From the CLI:

```bash
tradingagents backtest NVDA,AAPL --start 2026-06-01 --end 2026-08-01 --every 7
```

Each cell is scored on realized alpha against the instrument's regional benchmark, grouped by rating. Your own memory log is never written to, and re-running the same grid with `run_id=result.run_id` skips the cells that already ran, so an interrupted sweep continues where it stopped.

## Reproducibility

TradingAgents is LLM-driven, so two runs of the same ticker and date can differ. This is expected for a research tool built on language models, not a defect. The variation comes from a few distinct sources, and it helps to separate them.

Language model sampling is non-deterministic. Even at a fixed temperature, providers do not guarantee byte-identical output across calls, and reasoning models (the default GPT-6 family, and any thinking-mode model) vary the most because their internal reasoning is itself sampled.

Live data moves. News, StockTwits, and Reddit return different content as time passes, so a run today sees different inputs than a run last week even for the same historical trade date. Pin the analysis date to hold the price and indicator window fixed, but the social and news sources still reflect "now".

To reduce variation you can lower the sampling temperature. Set `temperature` in your config (or `TRADINGAGENTS_TEMPERATURE` in `.env`); lower values make models that honor it more repeatable. The current curated models are reasoning-first and largely ignore temperature, so for tighter reproducibility name a non-reasoning model in your config, or in `TRADINGAGENTS_DEEP_THINK_LLM` and `TRADINGAGENTS_QUICK_THINK_LLM`. Any model ID your provider serves is accepted, whether or not the picker lists it.

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["temperature"] = 0.0
# Reasoning models ignore temperature. For tighter reproducibility, name a
# non-reasoning model in deep_think_llm / quick_think_llm.
```

What does not vary anymore: the analyzed company identity is resolved deterministically from the ticker before any agent runs, and the market analyst grounds exact price and indicator claims in a verified data snapshot. Earlier reports of "different companies" or fabricated price levels across runs are addressed by these two mechanisms.

Backtest results are not guaranteed to match any published figure. Returns depend on the model, the temperature, the date range, data quality, and the sampling above. Treat the framework as a research scaffold for studying multi-agent analysis, not as a strategy with a fixed, replicable return.

## Contributing

Contributions are welcome: bug fixes, documentation, and feature ideas; past contributions are credited per release in [`CHANGELOG.md`](CHANGELOG.md).

## Citation

Please reference our work if you find *TradingAgents* provides you with some help :)

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```


### 本项目 fork 的决策角色与价格方案

交易员与研究经理、组合经理统一使用 deep 角色，分析师及辩论保持 quick；具体模型和推理强度由调用项目配置。交易员和最终决策结构有可选的参考价格/时点、建仓、加仓和减仓方案，旧结构仍可读取；方案首句固定为「区间 X–Y 美元（依据：具体价位）」或「不适用：原因」，随后包含触发/失效条件，缺少可靠报价时说明等待，不增加额外模型调用。三决策角色的可选 `target_allocation_pct` 统一以单标的标准仓位100%为单位：60表示用户计划量的六成，不是账户资产占比，也不是卖出现有持仓60%；目标随证据变化，不按评级固定映射。未知真实持仓和标准金额时不推算买卖数量。

分析师通过调用负载中的数据工具取证；Codex原生shell、文件和网络能力仍禁用。新闻分析师先调用get_news，输出事件表与催化剂日历。只有研究经理、交易员和组合经理输出评级/交易标签，分析师、多空和风险角色不输出交易结论。

三个决策角色引用同一五档定义。TraderProposal.action支持Buy、Overweight、Hold、Underweight、Sell；旧TraderAction导入作为五档别名保留。价格方案的ATR止损范围、推荐范围和最低盈亏比通过price_plan_*配置读取；未配置时分别使用1.0–2.5倍、1.5–2.0倍、1.5，买入按区间上沿计算。

角色提示按中短期决策收敛：市场报告先核验快照再查缺失指标；新闻先取数，情绪区分新闻语气与社交观点；多空只提供论据，三方风险负责具体价格方案审阅。组合经理收到市场关键价位表，默认沿用交易员点位，修改时说明理由。各角色采用提示词篇幅预算，不截断模型正文。

ETF与指数代理使用etf资产类型；个股使用stock。注入上下文末尾汇总编号的数据质量限制，各角色引用编号。运行结果保存实际含决策框架的注入文本，便于核对模型信息。

日线快照、指标与记忆结算共用可注册来源链，主项目配置Alpaca SIP复权→富途前复权→Yahoo复权；可用时末尾追加Alpha Vantage工具。缓存按来源隔离，截止日期与原缓存新鲜度、陈旧检测、价格补缺语义保留，快照及指标标出实际来源。

替代来源要求OpenD≥10.11：估值与内部人优先富途，新闻Alpaca→富途→Yahoo，报表SEC EDGAR→富途→Yahoo，宏观富途→FRED。Form144只作拟售；日历按周覆盖决策周期且只保留美国经济事件。VIX使用CBOE→FRED VIXCLS→Yahoo；板块手工→发行方持仓（含IWM）→Yahoo缓存。Yahoo首次连接/限频失败后批次熔断，HTTP单次≤10秒，下批重置。FRED可在config/secrets.env添加FRED_API_KEY，缺省不阻塞；仅剩未配置FRED时隐藏宏观工具。doctor核验OpenD服务端版本、FRED配置和CBOE。

新增price_anchors默认提供器：以完整P日为截止，复用stockstats输出EMA10、SMA20/50/200、ATR14、P日OHLC和20/60日高低点及日期。历史不足不外推，来源失败注明锚点不可用。扩展时段按同一P官方收盘计算，另列来源前收盘；偏差超过0.1%标记，缺锚点注明基准未核验。


### 来源观察钩子

`dataflows.vendor_observer.set_vendor_observer(callback)`按上下文隔离观察者，路由、日线、Yahoo请求/熔断及社交预取上报来源尝试和耗时；用返回的token调用`reset_vendor_observer`清理。未注册时不构造事件。观察者异常不改变取数结果；主项目负责净化并按类别汇总，fork不负责站点展示。

结算的来源查询也遵守运行配置`price_data_end_date`，不向SIP请求本日未完成K线；持有期和结束日排除口径保持。


### 分析期间补抓新闻（本项目维护分支）

`late_news_refresh`缺省为false。调用项目开启后，实时分析在研究经理与组合经理节点前复用`get_news`来源链，筛选上一次查询起点之后发布的新闻，按链接和标题去重，最多纳入10条；回放冻结窗口时跳过。补抓失败只记录数据限制。仅组合经理阶段新增的消息必须逐条评估对评级、目标配置与点位的影响，必要时写「建议重跑」，不得声称上游已评估。新状态字段为`news_last_fetched_at`、`late_news`及`late_news_errors`。

同一节点前还会调用调用项目通过私有配置`_late_macro_refresher`提供的取数函数，补抓当日经济数据标题。因数据源存在收录延迟，发布时间早于上次查询的标题也可能稍后才可见，所以经济数据不按发布时间筛选，而是与`_late_macro_seen`（初始上下文已有标题）及已纳入条目按标题去重，最多20条；未提供取数函数、关闭开关或回放时跳过。组合经理阶段新增的经济数据同样须逐条说明影响。新状态字段为`late_macro`与`late_macro_errors`。

### T31 概率生成与记忆重跑兼容

开启概率时研究经理各schema在分歧/引用核对后、评级前生成概率三字段，组合经理在评级前生成；关闭概率保留原schema JSON。给评级必须给0–1概率，薄证据向0.5收缩，仅关键输入缺失且评级也无法给出可写不可得原因；旧未知加载不补值。底层 `store_decision(..., replace_pending=True)` 仅供已核验成功live路径原子更新同日同标pending正文和评级；默认不覆盖，失败/不可用/REVIEW、历史backfill与已settled条目不覆盖、不重复计数。


### T17 按角色订阅CLI（本项目维护分支）

`role_llm_overrides`默认空、`role_llm_scheme`默认空、`legacy_speaker_rotation`默认false，不改变原quick/deep提示、schema和模型路径。覆盖仅允许bull、bear、research_manager、trader、aggressive、neutral、conservative、portfolio_manager，四分析师及未知角色配置立即报错。Claude仅接受`claude_exec / claude-opus-5-5 / high`；不允许全局Claude分析师provider。方案A按trade_date日期奇偶在bull/aggressive与bear/conservative间互换，方案B覆盖研究经理、交易员、组合经理；显式legacy轮换奇数日bear先发，structured仍保持双方并行opening/rebuttal。

runner直接启动本机Claude二进制，在临时空目录和清理API/provider凭据的环境先只读`auth status`。只有已登录claude.ai、firstParty及支持的订阅类型才发送prompt；Console/API、未登录、坏输出或认证超时不产生模型请求，也不修改登录或全局设置。模型调用关闭用户/项目settings、工具、MCP、自定义指令与持久会话；返回明确用户/项目自定义插件路径时拒绝，内置schema能力不作外部污染。订阅路径与实际model再次从控制输出校验，结构化结果本地验证，禁止API计费回落；显式角色包装允许按分类回退原默认Codex订阅模型。

记录每角色实际model、configured high及effective `NOT_REPORTED`（CLI未提供运行时字段），不保证服务端没有cap。成功和非零退出都保留返回tokens（含缓存）及目录价估算，缺字段标未知，不写零消费；订阅实际支付金额未知。额度/认证/配置失败不重试，超时/限频/传输按预算重试，默认1次、30秒退避，默认timeout300秒。离线错误分类桩不代表发生过真实额度事件；真实A/B须由调用项目显式测试入口隔离执行，不自动启用生产方案。

认证预检失败仅附安全role/stage/model_requests=0供调用项目区分尝试与实际调用，不创建模型用量行。失败控制输出中的actual_api_providers保留原firstParty/bedrock/vertex等安全路由字段，缺失未知，不能用配置订阅代替观察事实。


### T37 状态腿、持仓条件与R36（W1/W2）

`price_plan_legs` 默认true，动态 `LegTraderProposal` / `LegPortfolioDecision` 新增可选 `buy_legs` 和 `reduce_legs`，RM不增腿。所有新字段软归一，非数/非法枚举/类型置空并记录 `leg_validation_flags`，倒置区间/超过两腿保留并标记，内部标记不进入模型生成schema。买入状态为可执行、待触发、仅观察，减仓分超配回落与风险减配；加仓同价可以写同建仓，只补目标差额，目标内持仓不得因阻力受阻机械减仓，不连接账户。旧无腿记录不加执行段，新记录在减仓点位后逐腿显示执行条件。方向开关关闭且新腿已保存的记录允许无声明读取，不补造“否”。

`price_plan_target_rule` 默认r36。目标候选高于U+0.05ATR，1ATR内水平高点则该区间不合格；最近≥1ATR具名阻力为目标，完全无候选才U+3ATR，有近均线无合格远目标只观察。止损必须具名支撑减0–0.5ATR缓冲，仍从区间上沿量1–2.5ATR，最低RR1.5，禁止为RR倒推。`price_plan_legs=false` 且 `price_plan_target_rule=d1` 联合恢复旧提示/schema逐字，单独开关只恢复相应片段。新分支方向声明接受“是”加标点/空白及证据，异常保留并软flag；RM引用核对及概率在评级前、PM概率在评级前，62%归一0.62；旧严格声明与概率关闭快照保留。真实模型效果未调用验证（NOT_TESTED），逐腿结算由主项目后续W3实现。

T37角色回退默认 `role_llm_fallback=true`，仅在有角色覆盖的Claude角色启用。额度/配置错误立即开启批次熔断，暂态重试耗尽第二次开启；后续角色不调用Claude认证/模型。`_role_llm_breaker` 由调用项目每批新建并共享，线程安全计数及日志追加；结构化与自由文本共用包装，人工中止/非订阅分类错误照常抛出。回退按该角色原quick/deep模型与effort，复制默认runner仅绑定角色日志，不污染分析师实例，禁止Anthropic或其他APIprovider。空scheme/overrides保持原实例及空metadata。T23调用项目强制fallback=false/retries=0/timeout600/并发1。真实回退质量、耗时/额度及自然样本NOT_TESTED。

T37 W7/W8：Reddit结构取数失败不称0帖/无讨论，正常0帖以success来源状态和「正常（0条）」说明，minimum社交门槛继续在模型前跳过。EPS变化与惊喜由实际/基数原值重算并除abs(base)，abs(base)<0.05仅报美元差额和百分比不可比，不用来源极大surprisePercent覆盖近零规则。RM引用核对中文计字保留200门槛：汉字/标点各1，连续数字（含小数）/拉丁词元各1，空白不计；原始len与计字数同时记录，超长只标记不截断。提示/schema生成文字不受计字算法改动，旧联合开关快照保留。
