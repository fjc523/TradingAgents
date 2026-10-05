"""字符串兼容的社交查询结果；计数与可用状态直接来自取数路径。"""
import re


def live_social_reason(value):
    """仅由live展示调用，保留原原因元数据，不将正文作为原因。"""
    reason = getattr(value, 'reason', None) or getattr(value, 'source_reason', None)
    if not reason:
        return '抓取状态未提供'
    text = str(reason)
    status = re.search(r'(?i)HTTP\s*[:=]?\s*(\d{3})', text)
    if status:
        return 'HTTP ' + status.group(1)
    category = re.search(r'\b([A-Za-z]*(?:Error|Exception|Timeout))\b', text)
    if category:
        return category.group(1)
    return '来源获取失败'


class SocialResult(str):
    def __new__(cls, text, *, available=False, effective_posts=0):
        value=super().__new__(cls,text)
        value.available=available
        value.effective_posts=effective_posts
        return value


def social_absence_instruction(config=None):
    """K=0保留旧提示逐字；开启时未评估报告不能作为论据。"""
    if config is None:
        from tradingagents.dataflows.config import get_config
        config=get_config()
    return '\n情绪报告为“未评估”时不得作为论据，不得引用情绪分数或将新闻语气当独立社交证据。' if config.get('sentiment_min_social_posts',3)>0 else ''


def lesson_reference_instruction(config=None):
    """旧兼容配置不改变经理提示。"""
    if config is None:
        from tradingagents.dataflows.config import get_config
        config=get_config()
    return '' if config.get('lesson_min_settled_same_ticker',10)==0 and config.get('cross_ticker_lessons','stats')=='text' else '\n历史记录仅供参考；样本不足时不得据此改变方向判断。'
