"""第二模型CLI/角色绑定的离线契约，不发送真实请求。"""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pytest
from tradingagents.llm_clients.claude_exec.runner import (ClaudeExecRunner,clean_environment,build_argv,parse_output,
    ClaudeConfigError,ClaudeQuotaError,ClaudeTransientError,classify)
from tradingagents.graph.role_llms import resolve_overrides,build_role_llms,legacy_first,validate_role_config

@pytest.fixture(autouse=True)
def isolated_auth_status(monkeypatch):
    """本文件全部认证查询用桩，无真实CLI或网络。"""
    monkeypatch.setattr(subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,
        stdout=json.dumps({'loggedIn':True,'authMethod':'claude.ai','apiProvider':'firstParty','subscriptionType':'pro'})))


SCHEMA={'type':'object','properties':{'content':{'type':'string'}},'required':['content'],'additionalProperties':False}

def stdout(model='claude-opus-5-5',provider='firstParty',tools=None,content=None):
    return '\n'.join(json.dumps(e) for e in [dict(type='system',subtype='init',model=model,apiKeySource='none',tools=tools or ['StructuredOutput'],mcp_servers=[],skills=[]),
        dict(type='result',subtype='success',is_error=False,structured_output=content or {'content':'离线返回'},usage={'input_tokens':2,'cache_creation_input_tokens':100,'output_tokens':4},modelUsage={model:{'provider':provider}},total_cost_usd=.01)])

class Process:
    returncode=0
    def __init__(self,output=None,timeout=False,error=None):self.output=output or stdout();self.timeout=timeout;self.error=error
    def communicate(self,input,timeout):
        if self.timeout:raise subprocess.TimeoutExpired('fixture',timeout)
        if self.error:self.returncode=1;return '',self.error
        return self.output,''
    def poll(self):return self.returncode


def test_cli_direct_emptycwd_env_and_usage(tmp_path):
    seen=[]
    def popen(argv,**kwargs):
        assert not list(Path(kwargs['cwd']).iterdir())
        assert argv[0].endswith('/.local/bin/claude') and 'xagent' not in argv
        assert argv[argv.index('--model')+1]=='claude-opus-5-5'
        assert argv[argv.index('--effort')+1]=='high' and argv[argv.index('--tools')+1]==''
        assert argv[argv.index('--setting-sources')+1]=='' and '--safe-mode' in argv and '--restricted' in argv
        assert not {'ANTHROPIC_API_KEY','ANTHROPIC_AUTH_TOKEN','CLAUDE_CODE_OAUTH_TOKEN','AWS_ACCESS_KEY_ID'}&set(kwargs['env'])
        seen.append(argv);return Process()
    runner=ClaudeExecRunner(popen=popen,usage_log_path=str(tmp_path/'usage.jsonl'));runner.role='research_manager'
    result=runner.run('stdin prompt',SCHEMA)
    row=json.loads((tmp_path/'usage.jsonl').read_text())
    assert result.output=={'content':'离线返回'} and row['role']=='research_manager'
    assert row['model']=='claude-opus-5-5' and row['effective_effort']=='NOT_REPORTED'
    assert row['configured_effort']=='high' and row['tokens']['cache_creation_input_tokens']==100 and len(seen)==1


def test_env_allowlist_drops_api_routing_and_secret_proxy():
    env=clean_environment({'HOME':'fixture','ANTHROPIC_API_KEY':'secret','AWS_ACCESS_KEY_ID':'secret',
        'CLAUDE_CODE_USE_BEDROCK':'1','CLAUDE_CODE_OAUTH_TOKEN':'secret','ANTHROPIC_BASE_URL':'api',
        'HTTPS_PROXY':'http://user:secret@example','http_proxy':'http://127.0.0.1:7890','TRADINGAGENTS_MAX_DEBATE_ROUNDS':'99'})
    assert env['HOME']=='fixture' and 'HTTPS_PROXY' not in env and env['http_proxy'].endswith('7890')
    assert not any('KEY' in key or 'TOKEN' in key or key.startswith('TRADINGAGENTS_') for key in env)


@pytest.mark.parametrize('payload',[stdout(model='claude-sonnet'),stdout(provider='bedrock'),stdout(tools=['Read'])])
def test_output_actual_model_subscription_or_tools_mismatch_fails_closed(payload):
    with pytest.raises(ClaudeConfigError):parse_output(payload,SCHEMA,'claude-opus-5-5')


def test_schema_unknown_effort_and_failure_categories():
    with pytest.raises(ValueError):parse_output(stdout(content={'content':3}),SCHEMA,'claude-opus-5-5')
    assert parse_output(stdout(),SCHEMA,'claude-opus-5-5')[1]['effective_effort']=='NOT_REPORTED'
    assert classify('quota exceeded')=='quota' and classify('429 rate limit')=='rate_limit' and classify('',True)=='timeout'
    with pytest.raises(ClaudeQuotaError):ClaudeExecRunner(popen=lambda *a,**kw:Process(error='quota exceeded')).run('x',SCHEMA)
    with pytest.raises(ClaudeConfigError):ClaudeExecRunner(popen=lambda *a,**kw:Process(error='unknown model')).run('x',SCHEMA)
    with pytest.raises(ClaudeTransientError):ClaudeExecRunner(popen=lambda *a,**kw:Process(timeout=True),retries=0).run('x',SCHEMA)


def test_client_langchain_structured_uses_existing_validation():
    from tradingagents.llm_clients import create_llm_client
    model=create_llm_client('claude_exec','claude-opus-5-5').get_llm()
    model.runner._popen=lambda *a,**kw:Process()
    assert model.with_structured_output(SCHEMA).invoke('原prompt')=={'content':'离线返回'}
    with pytest.raises(ValueError):model.bind_tools([])


@pytest.mark.parametrize('role',['market','social','news','fundamentals','unknown'])
def test_analyst_and_unknown_override_rejected(role):
    with pytest.raises(ValueError):validate_role_config({'role_llm_overrides':{role:{'provider':'claude_exec','model':'claude-opus-5-5','effort':'high'}}})


def test_default_compatibility_and_even_odd_scheme():
    quick=object();deep=object()
    assert build_role_llms({},quick,deep)==({},{})
    assert set(resolve_overrides({'role_llm_scheme':'A','trade_date':'2026-10-02'}))=={'bull','aggressive'}
    assert set(resolve_overrides({'role_llm_scheme':'A','trade_date':'2026-10-01'}))=={'bear','conservative'}
    assert set(resolve_overrides({'role_llm_scheme':'B'}))=={'research_manager','trader','portfolio_manager'}
    assert legacy_first({'debate_mode':'legacy','legacy_speaker_rotation':True,'trade_date':'2026-10-01'})=='Bear Researcher'
    assert legacy_first({'debate_mode':'structured','legacy_speaker_rotation':True,'trade_date':'2026-10-01'})=='Bull Researcher'


@pytest.mark.parametrize('mode',['legacy','structured'])
def test_actual_graph_setup_binds_each_role_and_structured_side_both_phases(monkeypatch,mode):
    import tradingagents.graph.setup as setup
    import tradingagents.agents.researchers.structured_debate as structured
    from tradingagents.graph.conditional_logic import ConditionalLogic
    from tradingagents.graph.role_llms import ROLES
    quick=object();deep=object();models={role:object() for role in ROLES};seen={}
    factories={'create_bull_researcher':'bull','create_bear_researcher':'bear','create_research_manager':'research_manager',
        'create_trader':'trader','create_aggressive_debator':'aggressive','create_neutral_debator':'neutral',
        'create_conservative_debator':'conservative','create_portfolio_manager':'portfolio_manager'}
    def factory(role):
        def make(llm,*args):seen[role]=llm;return lambda state:{}
        return make
    for name,role in factories.items():monkeypatch.setattr(setup,name,factory(role))
    monkeypatch.setattr(setup,'create_market_analyst',factory('market'))
    phases=[]
    def turn(llm,side,phase):phases.append((llm,side,phase));return lambda state:{}
    monkeypatch.setattr(structured,'create_research_turn',turn)
    graph=setup.GraphSetup(quick,deep,ConditionalLogic(),20,config={'debate_mode':mode,'legacy_speaker_rotation':True,'trade_date':'2026-10-01'},role_llms=models).setup_graph(['market'])
    assert all(seen[role] is models[role] for role in ROLES) and seen['market'] is quick
    if mode=='structured':
        assert len(phases)==4 and all(llm is models[side] for llm,side,phase in phases)
        assert (('Bull Opening','Bear Opening'),'Research Rebuttals') in graph.waiting_edges
    else:assert (('Market Analyst',),'Bear Researcher') in graph.waiting_edges


def test_explicit_role_clients_model_provider_and_usage_role(monkeypatch):
    import tradingagents.graph.role_llms as role
    created=[]
    def client(provider,model,base_url,**kwargs):
        created.append((provider,model,kwargs));return SimpleNamespace(get_llm=lambda:object())
    monkeypatch.setattr(role,'create_llm_client',client)
    config={'llm_provider':'codex_exec','quick_think_llm':'gpt-6.1-sol','deep_think_llm':'gpt-6.1-sol',
        'codex_quick_reasoning_effort':'medium','codex_deep_reasoning_effort':'xhigh','role_llm_scheme':'B'}
    models,metadata=role.build_role_llms(config,object(),object())
    assert len(models)==8 and {row[2]['role'] for row in created}==set(role.ROLES)
    assert {key for key,value in metadata.items() if value['provider']=='claude_exec'}==role.DEEP_ROLES
    assert all(value['model']=='gpt-6.1-sol' and value['effort']=='medium' for key,value in metadata.items() if key not in role.DEEP_ROLES)


def test_transient_retry_budget_and_failed_schema_consumption_logged(tmp_path):
    attempts=[]
    def popen(*a,**kw):
        attempts.append(1);return Process(error='429 rate limit') if len(attempts)==1 else Process()
    runner=ClaudeExecRunner(popen=popen,retries=1,sleeper=lambda _:None,usage_log_path=str(tmp_path/'retry.jsonl'))
    runner.run('x',SCHEMA)
    rows=[json.loads(line) for line in (tmp_path/'retry.jsonl').read_text().splitlines()]
    assert len(attempts)==2 and [row['result'] for row in rows]==['rate_limit','success']
    assert rows[1]['retry']==1
    failed=ClaudeExecRunner(popen=lambda *a,**kw:Process(output=stdout(content={'content':3})),usage_log_path=str(tmp_path/'failed.jsonl'))
    with pytest.raises(ValueError):failed.run('x',SCHEMA)
    assert json.loads((tmp_path/'failed.jsonl').read_text())['tokens']['cache_creation_input_tokens']==100


@pytest.mark.parametrize('status',[{'loggedIn':True,'authMethod':'api_key','apiProvider':'firstParty','subscriptionType':'pro'},
    {'loggedIn':False,'authMethod':'none'}, {'loggedIn':True,'authMethod':'claude.ai','apiProvider':'bedrock','subscriptionType':'pro'},
    {'loggedIn':True,'authMethod':'claude.ai','apiProvider':'firstParty','subscriptionType':'free'}, 'broken-json', 'timeout'])
def test_auth_preflight_failures_make_zero_model_requests(status,tmp_path):
    queries=[]
    def auth(command,**kwargs):
        queries.append(command)
        assert command[-2:]==['auth','status'] and not list(Path(kwargs['cwd']).iterdir())
        assert 'ANTHROPIC_API_KEY' not in kwargs['env']
        if status=='timeout':raise subprocess.TimeoutExpired('auth',1)
        return SimpleNamespace(returncode=0,stdout=status if isinstance(status,str) else json.dumps(status))
    runner=ClaudeExecRunner(auth_run=auth,popen=lambda *a,**kw:pytest.fail('认证失败不能产生模型Popen'),usage_log_path=str(tmp_path/'usage.jsonl'))
    runner.role='research_manager'
    with pytest.raises(ClaudeConfigError,match='未发送模型请求') as failure:runner.run('prompt',SCHEMA)
    assert failure.value.role=='research_manager' and failure.value.stage=='auth_preflight' and failure.value.model_requests==0
    assert len(queries)==1 and not (tmp_path/'usage.jsonl').exists()


def test_auth_and_model_share_same_clean_environment_and_emptycwd():
    observed={}
    def auth(command,**kwargs):
        observed['auth']=kwargs
        return SimpleNamespace(returncode=0,stdout=json.dumps({'loggedIn':True,'authMethod':'claude.ai','apiProvider':'firstParty',
            'subscriptionType':'pro','email':'不要记录','accessToken':'不要记录'}))
    def model(command,**kwargs):
        assert kwargs['cwd']==observed['auth']['cwd'] and kwargs['env']==observed['auth']['env']
        assert not list(Path(kwargs['cwd']).iterdir())
        return Process()
    assert ClaudeExecRunner(auth_run=auth,popen=model).run('x',SCHEMA).output=={'content':'离线返回'}


@pytest.mark.parametrize('source',['user','project','local'])
def test_explicit_custom_plugin_fails_closed(source):
    events=[json.loads(line) for line in stdout().splitlines()]
    events[0]['plugins']=[{'name':'fixture-untrusted','path':'/tmp/fixture/plugin','source':source}]
    with pytest.raises(ClaudeConfigError):parse_output('\n'.join(json.dumps(row) for row in events),SCHEMA,'claude-opus-5-5')


def test_builtin_plugins_and_unknown_managed_remain_observation_limit():
    events=[json.loads(line) for line in stdout().splitlines()]
    events[0]['plugins']=[{'name':'cc-plugin-agents-md','path':'builtin:agents-md','source':'builtin'},
        {'name':'cc-plugin-plugin-authoring','path':'builtin:plugin-authoring','source':'builtin'}, {'source':'managed'}]
    assert parse_output('\n'.join(json.dumps(row) for row in events),SCHEMA,'claude-opus-5-5')[0]['content']


def test_nonzero_failure_preserves_consumption_and_actual_model(tmp_path):
    class Failed(Process):
        def communicate(self,input,timeout):
            self.returncode=1
            return stdout(),'quota exceeded'
    path=tmp_path/'failed.jsonl'
    with pytest.raises(ClaudeQuotaError):ClaudeExecRunner(popen=lambda *a,**kw:Failed(),usage_log_path=str(path)).run('x',SCHEMA)
    row=json.loads(path.read_text())
    assert row['tokens']['cache_creation_input_tokens']==100 and row['estimated_cost_usd']==.01
    assert row['model']=='claude-opus-5-5' and row['usage_status']=='REPORTED' and row['retry']==0


def test_failure_missing_usage_is_unknown_not_zero(tmp_path):
    path=tmp_path/'failed.jsonl'
    with pytest.raises(ClaudeQuotaError):ClaudeExecRunner(popen=lambda *a,**kw:Process(error='quota exceeded'),usage_log_path=str(path)).run('x',SCHEMA)
    row=json.loads(path.read_text())
    assert row['tokens'] is None and row['estimated_cost_usd'] is None and row['model'] is None
    assert row['usage_status']=='NOT_REPORTED'


@pytest.mark.parametrize('scope',['user','project','local'])
def test_plugin_descriptive_source_cannot_hide_explicit_scope(scope):
    events=[json.loads(line) for line in stdout().splitlines()]
    events[0]['plugins']=[{'name':'fixture-untrusted','source':'fixture-plugin@fixture-marketplace',
                           'scope':scope,'path':'/tmp/fixture/plugin'}]
    with pytest.raises(ClaudeConfigError):parse_output('\n'.join(json.dumps(row) for row in events),SCHEMA,'claude-opus-5-5')



def test_rejected_billing_route_failure_preserves_observed_provider(tmp_path):
    path=tmp_path/'failed.jsonl'
    with pytest.raises(ClaudeConfigError):ClaudeExecRunner(popen=lambda *a,**kw:Process(output=stdout(provider='bedrock')),usage_log_path=str(path)).run('x',SCHEMA)
    row=json.loads(path.read_text())
    assert row['actual_api_providers']==['bedrock'] and row['result']=='output_validation'
    assert row['tokens']['cache_creation_input_tokens']==100
