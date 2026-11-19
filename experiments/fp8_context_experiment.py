import json
import math
import time
from pathlib import Path

from src.inference import OLMoInferenceEngine
from transformers import AutoTokenizer

OUT_PATH = Path('data/processed/fp8_context_experiment.jsonl')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH = 'artifacts/OLMo-2-0425-1B-Instruct-FP8'
CONFIG = 'configs/inference_fp16.yaml'
PROMPT = 'Write a short greeting to a new user.'
PERCENTAGES = [0.10, 0.25, 0.50, 0.75, 1.0]
DEVICE = 'cuda'

# Load tokenizer to compute prompt length and model config for max positions
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
enc = tokenizer(PROMPT, return_tensors='pt')
prompt_len = enc.input_ids.shape[1]

# Try to read model max position from config file
import json as _json
cfg_path = Path(MODEL_PATH) / 'config.json'
if cfg_path.exists():
    cfg = _json.loads(cfg_path.read_text(encoding='utf-8'))
    max_pos = cfg.get('max_position_embeddings') or cfg.get('n_positions') or 4096
else:
    max_pos = 4096

available = max(max_pos - prompt_len, 0)
print(f'max_position_embeddings={max_pos}, prompt_len={prompt_len}, available={available}')

engine = OLMoInferenceEngine(config_path=CONFIG, model_path=MODEL_PATH, device=DEVICE)

for pct in PERCENTAGES:
    # Compute target new tokens as percentage of available
    target = max(1, math.floor(available * pct))
    # Ensure total tokens <= max_pos
    if prompt_len + target > max_pos:
        target = max_pos - prompt_len
    print(f'Running pct={pct*100:.0f}%, target_new_tokens={target}')

    start = time.time()
    res = engine.generate(prompt=PROMPT, max_new_tokens=target, disable_eos=True)
    end = time.time()

    entry = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': MODEL_PATH,
        'prompt': PROMPT,
        'prompt_len': prompt_len,
        'max_position_embeddings': max_pos,
        'percentage': pct,
        'requested_new_tokens': target,
        'generated_tokens': res.get('new_tokens_count') or res.get('response_token_count') or None,
        'generation_seconds': res.get('generation_seconds') or (end - start),
        'total_latency_seconds': res.get('total_latency_seconds') or (end - start),
        'tokens_per_sec': res.get('tokens_per_sec') or None,
        'generated_text_snippet': (res.get('generated_text') or '')[:1000],
        'full_text_length_chars': len(res.get('full_text') or ''),
    }

    with OUT_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Done: pct={pct*100:.0f}% tokens={entry['generated_tokens']} time={entry['generation_seconds']:.2f}s tok/s={entry['tokens_per_sec']}")

print('Experiment results saved to', OUT_PATH)
