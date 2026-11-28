import json
import logging
import urllib.request

from langchain_openai import ChatOpenAI

# Chat model for the LLM-driven single agent: an open model on a local OpenAI-compatible server (vLLM).

DEFAULT_LLM_URL = 'http://127.0.0.1:8000/v1'
DEFAULT_LLM_MODEL = 'Qwen/Qwen3-8B'


def build_llm(base_url=DEFAULT_LLM_URL, model=DEFAULT_LLM_MODEL, timeout=120):
    """LangChain chat model for the server at base_url. Raises if the server is down or does not serve model,
    so a run never starts without the LLM that is meant to drive it."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=10) as r:
            served = [m['id'] for m in json.load(r)['data']]
    except (OSError, ValueError, KeyError) as e:
        raise RuntimeError(f'LLM server {base_url} is not reachable ({type(e).__name__}: {e}); '
                           f'start it first (run.slurm does)') from e
    if model not in served:
        raise RuntimeError(f'LLM server {base_url} serves {served}, not {model!r}')
    # Evo2's vortex sets the root logger to INFO, so the openai client's httpx2 would log every LLM request
    logging.getLogger('httpx2').setLevel(logging.WARNING)
    # temperature 0: the same tool results give the same choice, so runs can be compared
    # enable_thinking False: Qwen3 otherwise writes a long <think> block before every tool call
    return ChatOpenAI(model=model, base_url=base_url, api_key='EMPTY', temperature=0, max_tokens=512,
                      timeout=timeout, max_retries=2,
                      extra_body={'chat_template_kwargs': {'enable_thinking': False}})
