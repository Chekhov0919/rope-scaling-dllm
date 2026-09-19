import argparse
import json
import os
import re

import exp_lib as E

_NUM = re.compile(r'\d')


def classify(prediction, reference, score):
    if score >= 100.0:
        return 'exact'
    a = re.sub(r'\s+', '', prediction)
    b = re.sub(r'\s+', '', reference)
    mx = max(len(a), len(b))
    sim = (1 - E.levenshtein_distance(a, b) / mx) if mx else 1.0
    if sim >= 0.6:
        return 'near'
    if prediction.strip() == '':
        return 'empty'
    if not _NUM.search(prediction):
        return 'no_number'
    return 'wrong_number'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('in_files', nargs='+')
    ap.add_argument('--group', default='method')
    ap.add_argument('--out', default='results/analysis.json')
    args = ap.parse_args()

    all_rows = []
    for path in args.in_files:
        if not os.path.isfile(path):
            print(f'[skip] missing {path}')
            continue
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for r in data.get('rows', []):
            r['_src'] = os.path.basename(path)
            r['_category'] = classify(r.get('prediction', ''),
                                      r.get('needle', ''),
                                      r.get('score', 0.0))
            all_rows.append(r)
    if not all_rows:
        raise SystemExit('no rows loaded')

    cats = ['exact', 'near', 'wrong_number', 'no_number', 'empty']
    summary = {'categories': cats, 'group_key': args.group, 'rows': []}
    groups = sorted({str(r.get(args.group)) for r in all_rows})
    for g in groups:
        sel = [r for r in all_rows if str(r.get(args.group)) == g]
        lens = sorted({r.get('context_tokens') for r in sel})
        print(f'--- {args.group} = {g} ---')
        header = f'{"len":>8}' + ''.join(f'{c:>12}' for c in cats) + f'{"n":>6}'

        print(header)
        for L in lens:
            cells = [r for r in sel if r.get('context_tokens') == L]
            counts = {c: sum(1 for r in cells if r['_category'] == c)
                      for c in cats}
            print(f'{L:>8}' + ''.join(f'{counts[c]:>12}' for c in cats)
                  + f'{len(cells):>6}')
        total = {c: sum(1 for r in sel if r['_category'] == c) for c in cats}
        print(f'{"TOTAL":>8}' + ''.join(f'{total[c]:>12}' for c in cats)
              + f'{len(sel):>6}')
        print()
        for r in sel:
            summary['rows'].append({
                args.group: g,
                'context_tokens': r.get('context_tokens'),
                'depth': r.get('depth'), 'sample': r.get('sample'),
                'steps': r.get('steps'), 'factor': r.get('factor'),
                'category': r['_category'], 'score': r.get('score'),
                'prediction': r.get('prediction')[:400],
            })

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f'[save] {args.out}  ({len(all_rows)} rows analyzed)')


if __name__ == '__main__':
    main()
