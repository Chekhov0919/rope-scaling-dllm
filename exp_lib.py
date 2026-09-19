import argparse
import hashlib
import json
import os
import re
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np  # noqa: E402
import torch  # noqa: E402

LLADA_CODE_DIRS = ['LLaDA-8B', 'LLaDA_8B']
LONGLLADA_ROOTS = ['LongLLaDA-main', 'LongLLaDA_main', 'LongLLaDA',
                   'longllada-main', 'longllada_main']
LONGLLADA_LLADA_DIRS = ['LongLLaDA-main/llada', 'LongLLaDA_main/llada']


def find_first(candidates, base='.'):
    for c in candidates:
        p = os.path.join(base, c) if base else c
        if os.path.isdir(p):
            return p
    return None


def find_first_file(candidates, base='.'):
    for c in candidates:
        p = os.path.join(base, c) if base else c
        if os.path.isfile(p):
            return p
    return None


def longllada_root():
    return find_first(LONGLLADA_ROOTS)


def resolve_llada_dirs():
    code_dir = find_first(LLADA_CODE_DIRS)
    if code_dir is None:
        raise FileNotFoundError('LLaDA code dir not found (looked for '
                                + ', '.join(LLADA_CODE_DIRS) + ')')
    ll_dir = find_first(LONGLLADA_LLADA_DIRS)
    if ll_dir is None:
        raise FileNotFoundError('LongLLaDA llada dir not found (looked for '
                                + ', '.join(LONGLLADA_LLADA_DIRS) + ')')
    return code_dir, ll_dir


def _add_code_dir_to_path(weights_dir):
    cands = [weights_dir]
    try:
        cands.append(resolve_llada_dirs()[0])
    except Exception:  # noqa: BLE001
        pass
    for d in cands:
        if (d and os.path.isdir(d)
                and os.path.isfile(os.path.join(d, 'configuration_llada.py'))
                and d not in sys.path):
            sys.path.insert(0, d)


def make_config(weights_dir, method, factor, **extra_cfg):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    cfg.scaling_method = method
    cfg.scaling_factor = float(factor)
    cfg.original_max_sequence_length = int(getattr(cfg, 'max_sequence_length', 4096))
    for k, v in extra_cfg.items():
        setattr(cfg, k, v)
    if method == 'ntk':
        cfg.rope_theta = float(cfg.rope_theta) * float(factor)
        cfg.scaling_factor = 1.0
    return cfg


def load_model(weights_dir, method='direct', factor=1.0,
               dtype='auto', device_map='auto', **extra_cfg):
    from transformers import AutoModelForCausalLM

    _add_code_dir_to_path(weights_dir)
    cfg = make_config(weights_dir, method, factor, **extra_cfg)
    if dtype == 'auto':
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        weights_dir, config=cfg, trust_remote_code=True,
        torch_dtype=dtype, device_map=device_map).eval()
    return model


def load_tokenizer(weights_dir=None):
    from transformers import AutoTokenizer

    code_dir, _ = resolve_llada_dirs()
    candidates = []
    for c in [weights_dir, code_dir, 'GSAI-ML/LLaDA-8B-Instruct']:
        if c and c not in candidates:
            candidates.append(c)
    last_err = None
    for c in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(c, trust_remote_code=True)
            tok.model_max_length = 1 << 22
            return tok
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f'No tokenizer found in {candidates}: {last_err}')


def rope_modules(model):
    out = []
    for m in model.modules():
        if (hasattr(m, 'get_rotary_embedding') and hasattr(m, 'rope_theta')
                and hasattr(m, 'config')):
            out.append(m)
    return out


def _rope_cache_dict(rot):
    for v in vars(rot).values():
        if isinstance(v, dict) and any(k.startswith('rope_') for k in v):
            return v
    return None


def clear_rope_cache(model):
    for rot in rope_modules(model):
        d = _rope_cache_dict(rot)
        if d is not None:
            for k in [k for k in d if k.startswith('rope_')]:
                d.pop(k, None)


def set_rope_attrs(model, **attrs):
    for rot in rope_modules(model):
        for k, v in attrs.items():
            setattr(rot, k, v)
    clear_rope_cache(model)


def _make_override_getter(orig):
    def wrapped(self, seq_len, device):
        ov = getattr(self, '_rope_override_fn', None)
        pos_fn = getattr(self, '_rope_pos_fn', None)
        if ov is None and pos_fn is None:
            return orig(self, seq_len, device)
        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            if ov is None:
                inv = _base_inv_freq(self, dim, device)
            else:
                inv = ov(self, dim, device)
            if pos_fn is None:
                seq = torch.arange(seq_len, device=device, dtype=torch.float)
            else:
                seq = pos_fn(seq_len, device)
            freqs = torch.einsum('i,j->ij', seq, inv)
            positions = torch.cat((freqs, freqs), dim=-1)
            return (positions.sin()[None, None, :, :],
                    positions.cos()[None, None, :, :])
    return wrapped


def install_position_remap(model, pos_fn, inv_fn=None):
    seen = set()
    for rot in rope_modules(model):
        cls = type(rot)
        if cls in seen:
            continue
        seen.add(cls)
        if not hasattr(cls, '_rope_orig_getter'):
            cls._rope_orig_getter = cls.get_rotary_embedding
            cls.get_rotary_embedding = _make_override_getter(cls._rope_orig_getter)
    for rot in rope_modules(model):
        rot._rope_pos_fn = pos_fn
        rot._rope_override_fn = inv_fn
    clear_rope_cache(model)


def chunk_reset_pos_fn(chunk, phase=0):
    def fn(seq_len, device):
        seq = torch.arange(seq_len, device=device, dtype=torch.float)
        return (seq + float(phase)) % float(chunk)
    return fn


def install_rope_override(model, override_fn):
    seen = set()
    for rot in rope_modules(model):
        cls = type(rot)
        if cls in seen:
            continue
        seen.add(cls)
        if not hasattr(cls, '_rope_orig_getter'):
            cls._rope_orig_getter = cls.get_rotary_embedding
            cls.get_rotary_embedding = _make_override_getter(cls._rope_orig_getter)
    for rot in rope_modules(model):
        rot._rope_override_fn = override_fn


def _base_inv_freq(rot, dim, device):
    t = torch.arange(0, dim, 2, device=device, dtype=torch.float)
    return 1.0 / (rot.rope_theta ** (t / dim))


RAW_ROPE_THETA = 500000.0


def assert_raw_theta(model, tol=1.01):
    for rot in rope_modules(model):
        theta = float(rot.rope_theta)
        if theta > RAW_ROPE_THETA * tol:
            raise ValueError(
                f'rope_theta={theta:g} looks pre-scaled (expected '
                f'{RAW_ROPE_THETA:g}). mode="bm" reasons about absolute '
                f'wavelengths and must run on a raw-theta model; install it '
                f'before any NTK prescaling.')


def band_override(rot, dim, device, scale=1.0, mode='full', split=0.5, power=2.0,
                  target=None, keep_high=0, s_max=1e9, low_tail=0, turns=1.0):
    inv = _base_inv_freq(rot, dim, device)
    n = inv.numel()
    if mode == 'full':
        return inv / scale
    idx = torch.arange(n, device=device, dtype=torch.float)
    if mode == 'ramp':
        lam = (2.0 * torch.pi) / inv
        tgt = float(target) if target else float(scale)
        t = float(turns) if turns else 1.0
        g = torch.clamp(tgt / (lam * t), min=1.0, max=float(s_max))
        return inv / g
    if mode == 'bm':
        lam = (2.0 * torch.pi) / inv
        tgt = float(target) if target else float(scale)
        g = torch.clamp(tgt / lam, min=1.0, max=float(s_max))
        if keep_high > 0:
            g[:int(keep_high)] = 1.0
        if low_tail > 0:
            g[n - int(low_tail):] = float(s_max)
        return inv / g
    if mode == 'low_only':
        w = (idx / n) >= (1.0 - split)
    elif mode == 'high_only':
        w = (idx / n) < split
    elif mode == 'smooth':
        g = 1.0 + (1.0 / scale - 1.0) * ((idx / (n - 1)) ** power)
        return inv * g
    else:
        raise ValueError(f'unknown band mode {mode}')
    g = torch.where(w, torch.full_like(idx, 1.0 / scale), torch.ones_like(idx))
    return inv * g


def install_band_override(model, scale=1.0, mode='full', split=0.5, power=2.0,
                          target=None, keep_high=0, s_max=1e9, low_tail=0,
                          turns=1.0):
    if mode in ('bm', 'ramp'):
        assert_raw_theta(model)
    def ov(rot, dim, device):
        return band_override(rot, dim, device, scale=scale, mode=mode,
                             split=split, power=power, target=target,
                             keep_high=keep_high, s_max=s_max, low_tail=low_tail,
                             turns=turns)
    install_rope_override(model, ov)


def clear_rope_override(model):
    install_rope_override(model, None)


ROMAN_PARAS = [
    "The history of the Roman Empire spans over a thousand years, beginning "
    "with the founding of Rome in 753 BC and ending with the fall of "
    "Constantinople in 1453 AD. The empire was one of the most powerful "
    "economic, cultural, political and military forces in the world of its "
    "time, and it remains a central subject of historical study to this day.",
    "During the height of the empire, Roman legions maintained a vast network "
    "of roads, aqueducts and fortifications that stretched from Britain in "
    "the west to Mesopotamia in the east. Trade routes connected the empire "
    "to distant lands, bringing silk from China, spices from India and gold "
    "from Africa into the markets of Rome itself.",
    "Roman law and governance evolved over centuries, from the early republic "
    "with its elected consuls and senate to the imperial system established "
    "under Augustus. The principle that all citizens were equal before the "
    "law, at least in theory, became one of the enduring legacies of Roman "
    "civilization.",
    "The Latin language spread across the empire and eventually gave rise to "
    "the Romance languages of modern Europe, including Italian, French, "
    "Spanish, Portuguese and Romanian. Latin also remained the language of "
    "scholarship, law and the Catholic Church for many centuries after the "
    "empire fell.",
    "Roman engineering achievements included concrete construction, the arch "
    "and the dome, which allowed the construction of monumental buildings "
    "such as the Colosseum, the Pantheon and the great baths. Many of these "
    "structures still stand today, a testament to the durability of Roman "
    "building techniques.",
    "The empire faced repeated challenges from barbarian invasions, economic "
    "decline and internal political instability. The western half of the "
    "empire collapsed in the fifth century AD, while the eastern half, known "
    "as the Byzantine Empire, continued for another thousand years.",
    "Roman culture absorbed and transformed the traditions of the Greeks, the "
    "Etruscans and the peoples of the Mediterranean world. Philosophy, "
    "literature, art and architecture all flourished under Roman patronage, "
    "blending native traditions with Hellenistic influences.",
    "The Roman army was a professional fighting force organized into legions "
    "of about five thousand men each. Soldiers served for twenty-five years "
    "and were rewarded with land and citizenship upon retirement, which "
    "helped to integrate conquered peoples into the empire.",
    "Religion in the Roman world was polytheistic, with a pantheon of gods "
    "and goddesses adapted from Greek mythology. Christianity, which arose in "
    "the eastern provinces, gradually spread throughout the empire and became "
    "the official religion in the late fourth century AD.",
    "Historians continue to debate the causes of the fall of the Roman "
    "Empire, citing factors as diverse as lead poisoning, climate change, "
    "plague, economic mismanagement and the sheer difficulty of governing "
    "such a vast territory with the technology of the ancient world.",
]

NEEDLE_TPL = "\nThe special magic number is {num}.\n"
QUESTION = ("What is the special magic number? "
            "Answer in the format 'The special magic number is X' where X "
            "is a number.")
INSTR_PREFIX = ("You are an intelligent AI assistant skilled in answering "
                "user questions.\nPlease keep your answers concise and "
                "clear. Do not talk about irrelevant topics or repeat your "
                "answers.\n")


def cell_seed(context_tokens, depth, sample, salt=''):
    key = f'{context_tokens}|{depth}|{sample}|{salt}'
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def build_needle_case(tokenizer, context_tokens, depth, sample,
                      num_range=(1000, 99999), tol=0.03, max_iter=8):
    rng = np.random.RandomState(cell_seed(context_tokens, depth, sample))
    num = int(rng.randint(*num_range))
    needle = NEEDLE_TPL.format(num=num)
    keyword = str(num)

    instr = (INSTR_PREFIX
             + 'The document given to you by the user is:\n{doc}\n\n'
             + 'Now, the question is: ' + QUESTION + '\n')

    def haystack_text(n_chars):
        out = []
        total = 0
        i = 0
        while total < n_chars:
            p = ROMAN_PARAS[i % len(ROMAN_PARAS)]
            out.append(p)
            total += len(p) + 1
            i += 1
        return '\n'.join(out)

    def build_prompt(doc_chars):
        hay = haystack_text(doc_chars)
        pos = int(len(hay) * depth)
        doc = hay[:pos] + needle + hay[pos:]
        return instr.format(doc=doc)

    n_chars = int(context_tokens * 5.0)
    prompt_text = build_prompt(n_chars)
    n_tokens = None
    for _ in range(max_iter):
        ids = tokenizer(prompt_text, add_special_tokens=False,
                        return_tensors='pt')
        n_tokens = int(ids['input_ids'].shape[1])
        if abs(n_tokens - context_tokens) <= context_tokens * tol:
            break
        n_chars = max(64, int(n_chars * context_tokens / max(1, n_tokens)))
        prompt_text = build_prompt(n_chars)

    return {
        'prompt_text': prompt_text,
        'needle': needle.strip(),
        'keyword': keyword,
        'n_tokens': n_tokens,
        'context_tokens': int(context_tokens),
        'depth': float(depth),
        'sample': int(sample),
    }


def run_generate(model, tokenizer, prompt_text, device='cuda',
                 steps=32, gen_length=32, block_length=32,
                 temperature=0.0, cfg_scale=0.0, remasking='low_confidence'):
    ll_dir = find_first(LONGLLADA_LLADA_DIRS)
    if ll_dir is None:
        ll_dir = resolve_llada_dirs()[1]
    if ll_dir not in sys.path:
        sys.path.insert(0, ll_dir)
    from llada_generate import generate  # noqa: PLC0415

    input_ids = tokenizer(prompt_text, add_special_tokens=False,
                          return_tensors='pt')['input_ids'].to(device)
    cfg = getattr(model, 'config', None)
    if cfg is not None:
        max_len = int(getattr(cfg, 'max_sequence_length', 0) or 0)
        if max_len and input_ids.shape[1] > max_len:
            print(f'[warn] prompt {input_ids.shape[1]} tokens > declared '
                  f'max_sequence_length {max_len}; RoPE positions are computed '
                  f'on the fly, continuing')
    with torch.no_grad():
        out = generate(model, input_ids, steps=steps, gen_length=gen_length,
                       block_length=block_length, temperature=temperature,
                       cfg_scale=cfg_scale, remasking=remasking)
    gen_ids = out[0, input_ids.shape[1]:]
    pred = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()
    return pred, gen_ids.tolist()


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1,
                           prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1]


def needlebench_score(prediction, reference, keyword):
    raw_prediction = prediction
    prediction = re.sub(r'\s+', '', prediction)
    reference = re.sub(r'\s+', '', reference)
    max_len = max(len(prediction), len(reference))
    edit_distance = levenshtein_distance(prediction, reference)
    score = 100 * (1 - edit_distance / max_len) if max_len else 100
    if keyword in raw_prediction:
        score = 100
    else:
        score = 0.2 * score
    return float(score)


LIB_VERSION = 3


def require_version(required):
    if LIB_VERSION != required:
        raise SystemExit(
            f'\n[FATAL] exp_lib.py interface version mismatch: '
            f'this script needs v{required}, the imported exp_lib.py is '
            f'v{LIB_VERSION}.\n'
            f'        Most likely exp_lib.py was not re-uploaded after the\n'
            f'        runner was updated. Re-upload exp_lib.py and retry.\n')
    print(f'[lib] exp_lib interface v{LIB_VERSION} OK')


def parse_variant(token):
    token = token.strip()
    if token == 'bm' or token.startswith('bm_kh'):
        keep_high, s_max = 8, 31.0
        if token.startswith('bm_kh'):
            m = re.match(r'bm_kh(\d+)_sm([0-9.]+)$', token)
            if not m:
                raise ValueError(
                    f'bad bm token {token!r}; use bm, or bm_kh<keep_high>_sm<s_max>')
            keep_high, s_max = int(m.group(1)), float(m.group(2))
        return ('band', 'direct', 1.0,
                {'bm_keep_high': keep_high, 'bm_s_max': s_max}, 'bm')
    if token.startswith('band_'):
        return ('band', 'direct', 1.0, {}, token[len('band_'):])
    if token == 'direct':
        return ('config', 'direct', 1.0, {}, None)
    if token.startswith('ntk_x'):
        return ('config', 'ntk', float(token.split('_x')[1]), {}, None)
    if token.startswith('pi_x'):
        return ('config', 'pi', float(token.split('_x')[1]), {}, None)
    if token.startswith('dyn_'):
        return ('config', 'dynamic_ntk', float(token.split('_')[1]), {}, None)
    if token.startswith('yarn_'):
        parts = token.split('_')
        if len(parts) != 4:
            raise ValueError(f'bad yarn token {token!r}; use yarn_<bf>_<bs>_<f>')
        bf, bs, f = float(parts[1]), float(parts[2]), float(parts[3])
        return ('config', 'yarn', f, {'yarn_beta_fast': bf,
                                      'yarn_beta_slow': bs}, None)
    raise ValueError(f'unknown variant token {token!r}')


def variant_label(kind, method, factor, band_mode, extra_cfg=None):
    if kind == 'band':
        if band_mode == 'bm':
            cfg = extra_cfg or {}
            return (f'bm_kh{cfg.get("bm_keep_high", 8)}'
                    f'_sm{cfg.get("bm_s_max", 31):g}')
        return f'band_{band_mode}'
    if method == 'direct':
        return 'direct'
    if method == 'yarn':
        cfg = extra_cfg or {}
        return (f'yarn_{cfg.get("yarn_beta_fast", 0):g}_'
                f'{cfg.get("yarn_beta_slow", 0):g}_{factor:g}')
    return f'{method}_x{factor:g}'


def parse_csv_float(s):
    return [float(x) for x in s.split(',') if x.strip() != '']


def parse_csv_int(s):
    return [int(x) for x in s.split(',') if x.strip() != '']


def save_results(path, rows, meta=None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {'meta': meta or {}, 'rows': rows}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f'[save] {path}  ({len(rows)} rows)')


def save_partial(out_path, rows, meta=None):
    if out_path.endswith('.json'):
        partial = out_path[:-5] + '.partial.json'
    else:
        partial = out_path + '.partial.json'
    save_results(partial, rows, meta=meta)


def print_table(rows, col_key, row_key, val_key, group_key=None):
    def mean(vals):
        return sum(vals) / len(vals) if vals else float('nan')

    groups = [None]
    if group_key:
        groups = sorted({r[group_key] for r in rows})
    for g in groups:
        sel = [r for r in rows if g is None or r[group_key] == g]
        if not sel:
            continue
        cols = sorted({r[col_key] for r in sel})
        row_vals = sorted({r[row_key] for r in sel})
        if g is not None:
            print(f'--- {group_key} = {g} ---')
        header = ' ' * 12 + ''.join(f'{str(c):>12}' for c in cols)
        print(header)
        for rv in row_vals:
            line = f'{str(rv):>12}'
            for c in cols:
                cell = mean([r[val_key] for r in sel
                             if r[row_key] == rv and r[col_key] == c])
                line += f'{cell:>12.1f}'
            print(line)
        print()


def add_common_args(parser):
    parser.add_argument('--weights', default=os.environ.get('LLADA_WEIGHTS', 'LLaDA-8B'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', default='auto')
    parser.add_argument('--gen_length', type=int, default=32)
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='results/exp.json')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--dry-run-chars', type=int, default=0)
    return parser


def runtime_estimate(rows, forward_passes):
    print(f'[plan] cells={len(rows)} total_forward_passes={forward_passes}')
