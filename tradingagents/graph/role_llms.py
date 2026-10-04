"""显式角色模型与日期方案；无配置时完全复用原quick/deep实例。"""
from datetime import date
from copy import deepcopy
from tradingagents.llm_clients import create_llm_client,build_llm_kwargs

ROLES=('bull','bear','research_manager','trader','aggressive','neutral','conservative','portfolio_manager')
DEEP_ROLES={'research_manager','trader','portfolio_manager'}

def validate_role_config(config):
    overrides=config.get('role_llm_overrides') or {}
    if not isinstance(overrides,dict) or set(overrides)-set(ROLES):raise ValueError('仅允许八个决策角色覆盖，禁止分析师/未知角色')
    for role,value in overrides.items():
        if not isinstance(value,dict) or set(value)!={'provider','model','effort'}:raise ValueError('角色覆盖必须仅含provider/model/effort')
        if value['provider']=='claude_exec':
            if value['model']!='claude-opus-5-5' or value['effort']!='high':raise ValueError('第二模型必须claude-opus-5-5 high')
        elif value['provider']=='codex_exec':
            if not isinstance(value['model'],str) or not value['model'] or value['effort'] not in ('low','medium','high','xhigh','max','ultra'):raise ValueError('Codex角色模型/effort无效')
        else:raise ValueError('角色provider须codex_exec或claude_exec')
    if config.get('role_llm_scheme') not in (None,'A','B'):raise ValueError('角色方案须空/A/B')
    if config.get('llm_provider')=='claude_exec':raise ValueError('Claude不可作为四分析师的全局provider')
    return overrides


def resolve_overrides(config):
    overrides=deepcopy(validate_role_config(config))
    scheme=config.get('role_llm_scheme')
    if scheme:
        if overrides:raise ValueError('角色方案与手工覆盖不能同时启用')
        even=date.fromisoformat(config['trade_date']).day%2==0 if scheme=='A' else False
        roles=(('bull','aggressive') if even else ('bear','conservative')) if scheme=='A' else ('research_manager','trader','portfolio_manager')
        overrides={role:{'provider':'claude_exec','model':'claude-opus-5-5','effort':'high'} for role in roles}
    return overrides


def legacy_first(config):
    if config.get('legacy_speaker_rotation') and config.get('debate_mode')=='legacy' and date.fromisoformat(config['trade_date']).day%2:
        return 'Bear Researcher'
    return 'Bull Researcher'


def build_role_llms(config, quick, deep, callbacks=None):
    overrides=resolve_overrides(config)
    if not overrides:return {},{}
    models={};metadata={}
    for role in ROLES:
        tier='deep' if role in DEEP_ROLES else 'quick'
        resolved=overrides.get(role,{'provider':config['llm_provider'],'model':config[f'{tier}_think_llm'],
            'effort':config.get(f'codex_{tier}_reasoning_effort') or config.get('codex_reasoning_effort') or 'high'})
        # 显式试验为每角色独立runner标usage，避免共用实例标签在并行中串位。
        role_config={**config,'llm_provider':resolved['provider'],f'codex_{tier}_reasoning_effort':resolved['effort']}
        kwargs=build_llm_kwargs(role_config,role=tier)
        kwargs['role']=role;kwargs['reasoning_effort']=resolved['effort']
        if callbacks:kwargs['callbacks']=callbacks
        for key in ('claude_binary','claude_timeout','claude_retries','claude_max_concurrency'):
            if key in config:kwargs[key]=config[key]
        client=create_llm_client(resolved['provider'],resolved['model'],config.get('backend_url'),**kwargs)
        models[role]=client.get_llm();metadata[role]={**resolved,'configured_effort':resolved['effort'],
            'effective_effort':'NOT_REPORTED' if resolved['provider']=='claude_exec' else '配置档位；实际用量日志分列'}
    return models,metadata
