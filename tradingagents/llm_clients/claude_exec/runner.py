"""订阅Claude CLI隔离执行；不继承API身份、不保存原始认证输出。"""
from datetime import UTC, datetime
from pathlib import Path
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from tradingagents.llm_clients.codex_exec import runner as common
from tradingagents.llm_clients.codex_exec.chat_model import _parse_structured, _validate_arguments
from tradingagents.llm_clients.errors import LLMNonRecoverableError


class ClaudeExecError(LLMNonRecoverableError):
    """不可自动切换模型或API计费的调用错误。"""


class ClaudeQuotaError(ClaudeExecError):
    """订阅额度不足。"""


class ClaudeConfigError(ClaudeExecError):
    """认证、模型或隔离配置失配。"""


class ClaudeTransientError(ClaudeExecError):
    """超时/限流/传输重试耗尽。"""


def clean_environment(source):
    """仅保留登录定位及无凭据代理/CA，认证由已登录本机订阅提供。"""
    names=('HOME','USER','LOGNAME','LANG','LC_ALL','LC_CTYPE','TMPDIR','SSL_CERT_FILE','SSL_CERT_DIR',
           'REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY',
           'http_proxy','https_proxy','all_proxy','no_proxy')
    env={key:source[key] for key in names if key in source}
    for key in list(env):
        if 'PROXY' in key.upper() and re.search(r'://[^/@]+:[^/@]+@',env[key]): del env[key]
    env.update(PATH='/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin',CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',
               DISABLE_TELEMETRY='1',DISABLE_ERROR_REPORTING='1')
    return env


def verify_subscription(binary, cwd, env, timeout, auth_run):
    """发prompt前只读校验订阅；不返回或记录账号/凭据及原始认证内容。"""
    try:
        status=auth_run([str(Path(binary).expanduser()),'auth','status'],cwd=cwd,env=env,
                        capture_output=True,text=True,timeout=min(float(timeout),30))
        value=json.loads(status.stdout)
    except (OSError,ValueError,subprocess.TimeoutExpired,TypeError) as exc:
        raise ClaudeConfigError('Claude订阅认证状态不可核验，未发送模型请求') from None
    if (status.returncode!=0 or not isinstance(value,dict) or value.get('loggedIn') is not True
            or value.get('authMethod')!='claude.ai' or value.get('apiProvider')!='firstParty'
            or value.get('subscriptionType') not in ('pro','max','team','enterprise')):
        raise ClaudeConfigError('Claude须已登录订阅，禁止Console/API计费；未发送模型请求')
    return {key:value[key] for key in ('loggedIn','authMethod','apiProvider','subscriptionType')}


def build_argv(binary, model, effort, schema):
    settings={'disableAllHooks':True,'autoMemoryEnabled':False,'disableClaudeAiConnectors':True,'enableAllProjectMcpServers':False}
    return [str(Path(binary).expanduser()),'--print','--safe-mode','--restricted','--setting-sources','',
        '--settings',json.dumps(settings),'--tools','','--mcp-config','{"mcpServers":{}}','--strict-mcp-config',
        '--no-session-persistence','--disable-slash-commands','--no-chrome','--permission-prompts','none',
        '--model',model,'--effort',effort,'--output-format','stream-json','--verbose','--json-schema',json.dumps(schema,ensure_ascii=False)]


def classify(text, timeout=False):
    if timeout:return 'timeout'
    value=text.lower()
    if any(x in value for x in ('usage limit','quota','out of extra usage','credit balance','insufficient credit')):return 'quota'
    if any(x in value for x in ('rate limit','rate_limit','429','too many requests')):return 'rate_limit'
    if any(x in value for x in ('model not found','invalid model','unknown model','not available for your','unauthorized','not logged in','authentication','401','unknown option')):return 'configuration'
    return 'transport'


def consumption_metadata(stdout):
    """失败退出也提取安全消费字段；缺失/坏JSON保持未知，不推断零成本。"""
    returned=[]
    for line in stdout.splitlines():
        try:
            value=json.loads(line)
            if isinstance(value,dict):returned.append(value)
        except (ValueError,TypeError):continue
    result=next((row for row in reversed(returned) if row.get('type')=='result'),{})
    usage=result.get('usage')
    models=result.get('modelUsage')
    names=list(models) if isinstance(models,dict) else []
    providers={row.get('provider') for row in models.values() if isinstance(row,dict) and row.get('provider') in ('firstParty','bedrock','vertex','anthropic')} if isinstance(models,dict) else set()
    return {'usage':usage if isinstance(usage,dict) else None,
            'actual_model':names[0] if len(names)==1 and names[0]=='claude-opus-5-5' else None,
            'estimated_cost_usd':result.get('total_cost_usd'),
            'actual_api_providers':sorted(providers) or None,
            'effective_effort':'NOT_REPORTED'}


def custom_plugin_reported(plugins):
    """仅明确用户/项目自定义来源拒绝；内置组件和未知managed状态不冒充污染。"""
    for plugin in plugins or []:
        if not isinstance(plugin,dict):continue
        sources={str(plugin.get(key,'')).lower() for key in ('source','scope')}
        path=str(plugin.get('path',''))
        if sources & {'user','project','local'} and path and not path.startswith('builtin:'):
            return True
    return False


def parse_output(stdout, schema, model):
    """只核实际控制消息model/provider，正文自称不构成来源证明。"""
    events=[json.loads(line) for line in stdout.splitlines() if line.strip()]
    init=next((event for event in events if event.get('type')=='system' and event.get('subtype')=='init'),{})
    result=next((event for event in reversed(events) if event.get('type')=='result'),{})
    models=list(result.get('modelUsage',{}))
    if init.get('model')!=model or not models or any(name!=model for name in models):
        raise ClaudeConfigError('Claude实际模型不匹配或缺失，拒绝替代模型')
    if init.get('apiKeySource')!='none' or any(row.get('provider')!='firstParty' for row in result['modelUsage'].values()):
        raise ClaudeConfigError('Claude订阅请求来源不匹配，禁止API计费回落')
    if (init.get('mcp_servers') or init.get('skills') or any(tool!='StructuredOutput' for tool in init.get('tools',[]))
            or custom_plugin_reported(init.get('plugins'))):
        raise ClaudeConfigError('Claude外部工具/MCP/skills隔离失配')
    if result.get('is_error') or result.get('subtype')!='success':
        raise ClaudeExecError('Claude返回错误结果')
    output=result.get('structured_output')
    if output is None:output=json.loads(result.get('result',''))
    output=_parse_structured(output,schema)
    _validate_arguments(output,schema,'Claude',schema.get('$defs',schema.get('definitions',{})))
    usage=result.get('usage',{})
    return output,{'usage':usage,'actual_model':model,'actual_api_providers':['firstParty'],'provider':'claude_exec','configured_effort':'high',
                   'effective_effort':'NOT_REPORTED','estimated_cost_usd':result.get('total_cost_usd'),
                   'cost_basis':'list估算，订阅实付未知'}


_LOG_LOCK=threading.Lock()
class ClaudeExecRunner:
    def __init__(self, *, model='claude-opus-5-5', reasoning_effort='high', binary='~/.local/bin/claude',
                 timeout=600,retries=0,max_concurrency=4,usage_log_path=None,popen=None,sleeper=time.sleep,auth_run=None):
        if model!='claude-opus-5-5' or reasoning_effort!='high':raise ClaudeConfigError('仅允许claude-opus-5-5 high既定选型')
        self.model=model;self.reasoning_effort=reasoning_effort;self.binary=binary
        self.timeout=float(timeout);self.retries=int(retries);self.max_concurrency=int(max_concurrency)
        if self.timeout<=0 or self.retries<0 or self.max_concurrency<1:raise ClaudeConfigError('Claude调用预算无效')
        self.usage_log_path=usage_log_path;self._popen=popen or subprocess.Popen;self._sleep=sleeper;self._auth_run=auth_run or subprocess.run;self.role=None

    def _record(self, events, result, attempt, duration):
        if not self.usage_log_path:return
        path=Path(self.usage_log_path).expanduser();path.parent.mkdir(parents=True,exist_ok=True)
        context=common._USAGE_CONTEXT.get()
        row={'timestamp':datetime.now(UTC).isoformat(),'ticker':context.get('ticker'),'role':self.role or context.get('role'),
             'call_type':context.get('call_type'),'provider':'claude_exec','model':events.get('actual_model'),
             'actual_api_providers':events.get('actual_api_providers'), 'requested_model':self.model,'reasoning_effort':self.reasoning_effort,'configured_effort':self.reasoning_effort,
             'effective_effort':events.get('effective_effort','NOT_REPORTED'),'tokens':events.get('usage'),'usage_status':'REPORTED' if isinstance(events.get('usage'),dict) else 'NOT_REPORTED',
             'estimated_cost_usd':events.get('estimated_cost_usd'),'cost_basis':'list估算，订阅实付未知',
             'result':result,'retry':attempt,'duration_seconds':duration,'warnings':[],'agent_actions':[]}
        with _LOG_LOCK,path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')

    def run(self,prompt,schema):
        common._acquire_slot(self.max_concurrency)
        try:
            for attempt in range(self.retries+1):
                if common._ABORT.is_set():raise ClaudeExecError('调用已中止')
                start=time.monotonic();category=None;events={}
                with tempfile.TemporaryDirectory(prefix='ta-claude-empty-') as cwd:
                    env=clean_environment(os.environ)
                    try:verify_subscription(self.binary,cwd,env,self.timeout,self._auth_run)
                    except ClaudeConfigError as exc:
                        exc.role=self.role;exc.stage='auth_preflight';exc.model_requests=0
                        raise
                    argv=build_argv(self.binary,self.model,self.reasoning_effort,schema)
                    try:
                        process=self._popen(argv,cwd=cwd,env=env,stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
                    except OSError as exc:raise ClaudeConfigError('Claude本机二进制不可启动') from exc
                    with common._ACTIVE_LOCK:common._ACTIVE.add(process)
                    try:
                        try:stdout,stderr=process.communicate(input=prompt,timeout=self.timeout)
                        except subprocess.TimeoutExpired:
                            common._terminate(process);category='timeout';stdout=stderr=''
                    finally:
                        with common._ACTIVE_LOCK:common._ACTIVE.discard(process)
                    if common._ABORT.is_set():raise ClaudeExecError('调用已中止')
                    events=consumption_metadata(stdout)
                    if category is None and process.returncode==0:
                        try:output,events=parse_output(stdout,schema,self.model)
                        except Exception:
                            self._record(events,'output_validation',attempt,time.monotonic()-start)
                            raise
                        self._record(events,'success',attempt,time.monotonic()-start)
                        return common.CodexResult(output,events,argv)
                    category=category or classify(stderr+'\n'+stdout)
                    self._record(events,category,attempt,time.monotonic()-start)
                    if category=='quota':raise ClaudeQuotaError('Claude订阅额度不足')
                    if category=='configuration':raise ClaudeConfigError('Claude配置/认证/模型不可用')
                if attempt<self.retries:self._sleep(min(30*(attempt+1),120))
            raise ClaudeTransientError('Claude '+str(category)+' 重试耗尽')
        finally:common._release_slot()
