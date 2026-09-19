import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import argparse
import gc

import torch

import exp_lib as E


def parse_args():
    ap = argparse.ArgumentParser()
    E.add_common_args(ap)
    ap.add_argument('--lengths', default='8192,16384,24576')
    ap.add_argument('--depths', default='0.5')
    ap.add_argument('--samples', type=int, default=5)
    ap.add_argument('--scale', type=float, default=14.0)
    ap.add_argument('--band-modes', default='full,low_only,high_only,smooth')
    ap.add_argument('--yarn-cfgs', default='32:1:8,8:1:8,4:1:16')
    ap.add_argument('--split', type=float, default=0.5)
    ap.add_argument('--power', type=float, default=2.0)
    ap.add_argument('--bm-keep-high', default='0,8,16,32')
    ap.add_argument('--bm-smax', default='14,31,55')
    ap.add_argument('--bm-target', type=float, default=0.0)
    ap.add_argument('--ramp-smax', default='')
    ap.add_argument('--ramp-turns', default='1,2')
    ap.add_argument('--ramp-target', type=float, default=0.0)
    ap.add_argument('--chunks', default='')
    ap.add_argument('--chunk-phases', default='')
    ap.add_argument('--flash', action='store_true')
    args = ap.parse_args()
    args.out = args.out if args.out != 'results/exp.json' else 'results/expB2_freqband.json'
    return args


def parse_yarn_cfg(token):
    parts = token.split(':')
    if len(parts) != 3:
        raise ValueError(f'yarn cfg must be beta_fast:beta_slow:factor, got {token!r}')
    return float(parts[0]), float(parts[1]), float(parts[2])


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run(args):
    torch.manual_seed(args.seed)
    lengths = E.parse_csv_int(args.lengths)
    depths = E.parse_csv_float(args.depths)
    band_modes = [m.strip() for m in args.band_modes.split(',') if m.strip()]
    yarn_cfgs = [parse_yarn_cfg(t) for t in args.yarn_cfgs.split(',') if t.strip()]
    tokenizer = E.load_tokenizer(args.weights)
    extra = {'flash_attention': True} if args.flash else {}

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
    meta = {'purpose': 'B2 freq-band controls + YaRN tuning',
            'scale': args.scale, 'band_modes': band_modes, 'yarn_cfgs': yarn_cfgs,
            'lengths': lengths, 'depths': depths, 'samples': args.samples,
            'seed': args.seed, 'weights': args.weights}

    def score_cells(model, label, spec, only_length=None):
        for cell in plan:
            if only_length is not None and cell['context_tokens'] != only_length:
                continue
            case = cell['build']()
            pred, gen_ids = E.run_generate(
                model, tokenizer, case['prompt_text'], device=args.device,
                steps=32, gen_length=32, block_length=32,
                temperature=0.0, cfg_scale=0.0, remasking='low_confidence')
            score = E.needlebench_score(pred, case['needle'], case['keyword'])
            rows.append({
                'variant': label, 'spec': spec,
                'context_tokens': cell['context_tokens'],
                'depth': cell['depth'], 'sample': cell['sample'],
                'n_tokens': case['n_tokens'],
                'needle': case['needle'], 'keyword': case['keyword'],
                'prediction': pred, 'score': score,
            })
        E.save_partial(args.out, rows, meta=meta)

    print('\n[load] raw-theta model (direct / band / yarn)')
    model = E.load_model(args.weights, method='direct', factor=1.0,
                         dtype=args.dtype, **extra)

    print('[run] reference direct')
    score_cells(model, 'direct', {})

    for mode in band_modes:
        print(f'[run] band mode={mode} scale={args.scale}')
        E.install_band_override(model, scale=args.scale, mode=mode,
                                split=args.split, power=args.power)
        score_cells(model, f'band_{mode}', {'scale': args.scale})
    E.clear_rope_override(model)

    bm_keep = E.parse_csv_int(args.bm_keep_high)
    bm_smax = E.parse_csv_float(args.bm_smax)
    if bm_keep and bm_smax:
        for ct in lengths:
            tgt = args.bm_target if args.bm_target > 0 else float(ct)
            for kh in bm_keep:
                for sm in bm_smax:
                    label = f'bm_kh{kh}_sm{sm:g}'
                    print(f'[run] {label} target={tgt:g}')
                    E.install_band_override(model, mode='bm', target=tgt,
                                            keep_high=kh, s_max=sm)
                    score_cells(model, label,
                                {'mode': 'bm', 'target': tgt,
                                 'keep_high': kh, 's_max': sm},
                                only_length=ct)
        E.clear_rope_override(model)

    ramp_smax = E.parse_csv_float(args.ramp_smax)
    ramp_turns = E.parse_csv_float(args.ramp_turns)
    if ramp_smax:
        for ct in lengths:
            tgt = args.ramp_target if args.ramp_target > 0 else float(ct)
            for sm in ramp_smax:
                for tn in ramp_turns:
                    label = f'ramp_sm{sm:g}_t{tn:g}'
                    print(f'[run] {label} target={tgt:g}')
                    E.install_band_override(model, mode='ramp', target=tgt,
                                            s_max=sm, turns=tn)
                    score_cells(model, label,
                                {'mode': 'ramp', 'target': tgt,
                                 's_max': sm, 'turns': tn},
                                only_length=ct)
        E.clear_rope_override(model)

    chunks = E.parse_csv_int(args.chunks)
    if chunks:
        phases = E.parse_csv_int(args.chunk_phases)
        for C in chunks:
            ph = phases if phases else [C // 4]
            for p in ph:
                label = f'chunk{C}_p{p}'
                print(f'[run] {label}')
                E.install_position_remap(model, E.chunk_reset_pos_fn(C, p))
                score_cells(model, label, {'chunk': C, 'phase': p})
        E.install_position_remap(model, None)

    for bf, bs, f in yarn_cfgs:
        print(f'[run] yarn beta_fast={bf} beta_slow={bs} factor={f}')
        E.set_rope_attrs(model, scaling_method='yarn', scaling_factor=f,
                         yarn_beta_fast=bf, yarn_beta_slow=bs)
        score_cells(model, 'yarn', {'beta_fast': bf, 'beta_slow': bs,
                                    'factor': f})

    free_model(model)

    print(f'\n[load] ntk reference model (lambda={args.scale:g})')
    model = E.load_model(args.weights, method='ntk', factor=args.scale,
                         dtype=args.dtype, **extra)
    print(f'[run] reference ntk_x{args.scale:g}')
    score_cells(model, f'ntk_x{args.scale:g}', {'scale': args.scale})
    free_model(model)

    E.print_table(rows, col_key='context_tokens', row_key='variant',
                  val_key='score')
    E.save_results(args.out, rows, meta=meta)


if __name__ == '__main__':
    run(parse_args())
