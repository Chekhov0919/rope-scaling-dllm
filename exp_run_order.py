import argparse
import json
import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import gc

import numpy as np
import torch

import exp_lib as E

E.require_version(3)

NEEDLE_TPL = "\nThe special magic number is {num}.\n"
QUESTION = {
    'first': ("Which magic number appears first in the document? Answer in the "
              "format 'The first magic number is X' where X is a number."),
    'last': ("Which magic number appears last in the document? Answer in the "
             "format 'The last magic number is X' where X is a number."),
}
INSTR = ("You are an intelligent AI assistant skilled in answering user "
         "questions.\nPlease keep your answers concise and clear. Do not talk "
         "about irrelevant topics or repeat your answers.\n")


def parse_args():
    ap = argparse.ArgumentParser()
    E.add_common_args(ap)
    ap.set_defaults(seed=2025)
    ap.add_argument('--variants',
                    default='direct,ntk_x14,ntk_x55,band_low_only,pi_x8')
    ap.add_argument('--lengths', default='8192')
    ap.add_argument('--questions', default='first,last')
    ap.add_argument('--samples', type=int, default=5)
    ap.add_argument('--needles', type=int, default=3)
    ap.add_argument('--needle-depths', default='0.15,0.5,0.85')
    ap.add_argument('--band-scale', type=float, default=14.0)
    ap.add_argument('--band-split', type=float, default=0.5)
    ap.add_argument('--band-power', type=float, default=2.0)
    ap.add_argument('--flash', action='store_true')
    ap.add_argument('--report-confusion', action='store_true', default=True)
    args = ap.parse_args()
    args.out = args.out if args.out != 'results/exp.json' else 'results/expOrder.json'
    return args


parse_variant = E.parse_variant


def build_order_case(tokenizer, context_tokens, sample, question,
                     needle_depths, num_range=(1000, 99999), tol=0.03,
                     max_iter=8):
    rng = np.random.RandomState(
        E.cell_seed(context_tokens, 0.0, sample, salt=f'order-{question}'))
    numbers = []
    while len(numbers) < len(needle_depths):
        candidate = int(rng.randint(*num_range))
        text = str(candidate)
        if any(text in str(n) or str(n) in text for n in numbers):
            continue
        numbers.append(candidate)

    def haystack_text(n_chars):
        out, total, i = [], 0, 0
        while total < n_chars:
            p = E.ROMAN_PARAS[i % len(E.ROMAN_PARAS)]
            out.append(p)
            total += len(p) + 1
            i += 1
        return '\n'.join(out)

    def build_prompt(n_chars):
        hay = haystack_text(n_chars)
        doc = hay
        for depth, num in sorted(zip(needle_depths, numbers), reverse=True):
            pos = int(len(hay) * depth)
            doc = doc[:pos] + NEEDLE_TPL.format(num=num) + doc[pos:]
        return (INSTR + 'The document given to you by the user is:\n' + doc
                + '\n\nNow, the question is: ' + QUESTION[question] + '\n')

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

    expected = numbers[0] if question == 'first' else numbers[-1]
    return {
        'prompt_text': prompt_text,
        'needle': NEEDLE_TPL.format(num=expected).strip(),
        'keyword': str(expected),
        'numbers': numbers,
        'needle_depths': list(needle_depths),
        'question': question,
        'n_tokens': n_tokens,
        'context_tokens': int(context_tokens),
        'sample': int(sample),
    }


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def confusion_rank(prediction, numbers):
    hits = [i for i, num in enumerate(numbers) if str(num) in prediction]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return 'multiple'
    return 'none'


def run(args):
    torch.manual_seed(args.seed)
    lengths = E.parse_csv_int(args.lengths)
    questions = [q.strip() for q in args.questions.split(',') if q.strip()]
    needle_depths = E.parse_csv_float(args.needle_depths)
    variants = [parse_variant(v) for v in args.variants.split(',') if v.strip()]
    tokenizer = E.load_tokenizer(args.weights)

    prompt_cache = {}

    def case_for(length, sample, question):
        key = (length, sample, question)
        if key not in prompt_cache:
            prompt_cache[key] = build_order_case(
                tokenizer, length, sample, question, needle_depths)
        return prompt_cache[key]

    rows = []
    meta = {'purpose': 'multi-needle ordering probe', 'variants': args.variants,
            'lengths': lengths, 'questions': questions, 'samples': args.samples,
            'needles': args.needles, 'needle_depths': needle_depths,
            'seed': args.seed, 'weights': args.weights}

    if args.dry_run:
        for length in lengths:
            for question in questions:
                for sample in range(args.samples):
                    trial = args.dry_run_chars or min(256, length)
                    case = build_order_case(tokenizer, trial, sample, question,
                                            needle_depths)
                    print(f'[dry] {length:>6} {question:<5} sample={sample} '
                          f'n_tokens={case["n_tokens"]} numbers={case["numbers"]} '
                          f'answer={case["keyword"]} '
                          f'ranks={[confusion_rank(str(n), case["numbers"]) for n in case["numbers"]]}')
        print(f'[dry] variants={[E.variant_label(v[0], v[1], v[2], v[4], v[3]) for v in variants]}')
        print(f'[dry] cells={len(lengths) * len(questions) * args.samples} '
              f'x {len(variants)} variants')
        return

    for kind, method, factor, extra_cfg, band_mode in variants:
        label = E.variant_label(kind, method, factor, band_mode, extra_cfg)
        print(f'\n[load] variant={label}')
        model_kw = {k: v for k, v in extra_cfg.items()
                    if not k.startswith('bm_')}
        if args.flash:
            model_kw['flash_attention'] = True
        model = E.load_model(args.weights, method=method, factor=factor,
                             dtype=args.dtype, **model_kw)
        for length in lengths:
            if kind == 'band':
                if band_mode == 'bm':
                    E.install_band_override(
                        model, mode='bm', target=float(length),
                        keep_high=extra_cfg.get('bm_keep_high', 8),
                        s_max=extra_cfg.get('bm_s_max', 31.0))
                else:
                    E.install_band_override(model, scale=args.band_scale,
                                            mode=band_mode,
                                            split=args.band_split,
                                            power=args.band_power)
            for question in questions:
                for sample in range(args.samples):
                    case = case_for(length, sample, question)
                    pred, _ = E.run_generate(
                        model, tokenizer, case['prompt_text'], device=args.device,
                        steps=32, gen_length=32, block_length=32,
                        temperature=0.0, cfg_scale=0.0,
                        remasking='low_confidence')
                    score = E.needlebench_score(pred, case['needle'],
                                                case['keyword'])
                    rows.append({
                        'variant': label, 'method': method, 'factor': factor,
                        'context_tokens': length, 'question': question,
                        'sample': sample, 'n_tokens': case['n_tokens'],
                        'numbers': case['numbers'],
                        'needle_depths': case['needle_depths'],
                        'keyword': case['keyword'], 'prediction': pred,
                        'score': score,
                        'emitted_rank': confusion_rank(pred, case['numbers']),
                    })
            E.save_partial(args.out, rows, meta=meta)
        free_model(model)

    E.print_table(rows, col_key='question', row_key='variant', val_key='score',
                  group_key='context_tokens')

    if args.report_confusion:
        print('\n[confusion] rank of the emitted needle (0 = first, '
              f'{args.needles - 1} = last)')
        cells = {}
        for r in rows:
            key = (r['variant'], r['context_tokens'], r['question'])
            d = cells.setdefault(key, {})
            d[r['emitted_rank']] = d.get(r['emitted_rank'], 0) + 1
        for key in sorted(cells, key=lambda k: (k[1], k[0], k[2])):
            dist = ', '.join(f'{k}:{v}' for k, v in sorted(
                cells[key].items(), key=lambda kv: str(kv[0])))
            print(f'  {key[1]//1024}K {key[0]:<16} {key[2]:<5} -> {dist}')

    E.save_results(args.out, rows, meta=meta)


if __name__ == '__main__':
    run(parse_args())
