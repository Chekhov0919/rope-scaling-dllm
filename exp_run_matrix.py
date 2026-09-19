import argparse

import torch

import exp_lib as E

E.require_version(3)


def parse_args():
    ap = argparse.ArgumentParser()
    E.add_common_args(ap)
    ap.set_defaults(seed=2025)
    ap.add_argument('--lengths', default='2048,4096,8192,16384')
    ap.add_argument('--depths', default='0,0.25,0.5,0.75,1.0')
    ap.add_argument('--samples', type=int, default=5)
    ap.add_argument('--methods', default='direct,ntk_x14,pi_x8,yarn_32_1_8,dyn_16')
    ap.add_argument('--flash', action='store_true')
    args = ap.parse_args()
    args.out = args.out if args.out != 'results/exp.json' else 'results/expM_matrix.json'
    return args


def parse_method(token):
    token = token.strip()
    if token == 'direct':
        return ('direct', 1.0, {})
    if token.startswith('ntk_x'):
        return ('ntk', float(token.split('_x')[1]), {})
    if token.startswith('pi_x'):
        return ('pi', float(token.split('_x')[1]), {})
    if token.startswith('dyn_'):
        return ('dynamic_ntk', float(token.split('_')[1]), {})
    if token.startswith('yarn_'):
        parts = token.split('_')
        if len(parts) != 4:
            raise ValueError(f'bad yarn token {token!r}; use yarn_<bf>_<bs>_<f>')
        bf, bs, f = float(parts[1]), float(parts[2]), float(parts[3])
        return ('yarn', f, {'yarn_beta_fast': bf, 'yarn_beta_slow': bs})
    raise ValueError(f'unknown method token {token!r}')


def run(args):
    torch.manual_seed(args.seed)
    lengths = E.parse_csv_int(args.lengths)
    depths = E.parse_csv_float(args.depths)
    methods = [parse_method(m) for m in args.methods.split(',') if m.strip()]
    tokenizer = E.load_tokenizer(args.weights)

    prompt_cache = {}
    plan = []
    for ct in lengths:
        for d in depths:
            for s in range(args.samples):
                key = (ct, d, s)
                def build(k=key):
                    if k not in prompt_cache:
                        prompt_cache[k] = E.build_needle_case(tokenizer, *k)
                    return prompt_cache[k]
                plan.append({'context_tokens': ct, 'depth': d, 'sample': s,
                             'build': build})

    rows = []
    meta = {'purpose': 'unified method matrix (R1/R2)', 'methods': args.methods,
            'lengths': lengths, 'depths': depths, 'samples': args.samples,
            'seed': args.seed, 'weights': args.weights}
    for method, factor, extra in methods:
        print(f'\n[load] method={method} factor={factor} extra={extra}')
        extra_cfg = dict(extra)
        if args.flash:
            extra_cfg['flash_attention'] = True
        model = E.load_model(args.weights, method=method, factor=factor,
                             dtype=args.dtype, **extra_cfg)
        for cell in plan:
            case = cell['build']()
            pred, gen_ids = E.run_generate(
                model, tokenizer, case['prompt_text'], device=args.device,
                steps=32, gen_length=32, block_length=32,
                temperature=0.0, cfg_scale=0.0, remasking='low_confidence')
            score = E.needlebench_score(pred, case['needle'], case['keyword'])
            rows.append({
                'method': method, 'factor': float(factor),
                'spec': method_label(method, factor),
                'context_tokens': cell['context_tokens'],
                'depth': cell['depth'], 'sample': cell['sample'],
                'n_tokens': case['n_tokens'],
                'needle': case['needle'], 'keyword': case['keyword'],
                'prediction': pred, 'score': score,
            })
        del model
        torch.cuda.empty_cache()
        E.save_partial(args.out, rows, meta=meta)

    E.print_table(rows, col_key='context_tokens', row_key='method',
                  val_key='score', group_key='depth')
    E.save_results(args.out, rows, meta=meta)


def method_label(method, factor):
    if method == 'ntk':
        return f'ntk_x{factor:g}'
    if method == 'pi':
        return f'pi_x{factor:g}'
    if method == 'dynamic_ntk':
        return f'dyn_{factor:g}'
    return method


if __name__ == '__main__':
    run(parse_args())
