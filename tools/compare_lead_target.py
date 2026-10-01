#!/usr/bin/env python3
# 사용: python3 ~/rank_compare.py results/lead_on_0930_rank 07_170530:3:102 09_160729:19:99 11_190205:2:73 13_191112:15:<충돌> ...
import sys, csv, os
out = sys.argv[1]
def load(path): return list(csv.DictReader(open(path)))
print(f"{'영상':10} {'조합':5} {'대상첫LEAD':>9} {'대상첫DANGER':>11} {'오대상DANGER구간(충돌전)':>24} {'영상첫DANGER':>10} {'판정':>10}")
for spec in sys.argv[2:]:
    vid, target, collision = spec.split(':'); collision = int(collision)
    for tag in ('base', 'rank'):
        sel = load(os.path.join(out, f'{vid}_{tag}_lead_select.csv'))
        ban = load(os.path.join(out, f'{vid}_{tag}_banner.csv'))
        first_lead = next((int(r['frame']) for r in sel if r['trackId'] == target and r['isLead'] == '1'), None)
        first_danger_target = next((int(r['frame']) for r in ban if r['bannerLevel'] == 'DANGER' and r['activeLeadId'] == target), None)
        wrong = [int(r['frame']) for r in ban if r['bannerLevel'] == 'DANGER' and r['activeLeadId'] not in (target, '-1') and int(r['frame']) < collision]
        segs = []
        for f in wrong:
            if segs and f == segs[-1][1] + 1: segs[-1][1] = f
            else: segs.append([f, f])
        first_any = next((int(r['frame']) for r in ban if r['bannerLevel'] == 'DANGER'), None)
        verdict = 'OK' if first_danger_target is not None and first_danger_target < collision else ('WRONG-ONLY' if wrong else 'MISS')
        print(f"{vid:10} {tag:5} {str(first_lead):>9} {str(first_danger_target):>11} {','.join(f'{a}~{b}' for a,b in segs) or '-':>24} {str(first_any):>10} {verdict:>10}")