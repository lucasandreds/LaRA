import json
import numpy as np

def jsonl(p):   return {r['scene']: r for r in map(json.loads, open(p))}
def metr(p):    d = json.load(open(p)); return dict(zip(d['name'], d['psnr']))

KEEP = {0: 'outputs/metrics/gso_rot0.json', 5: 'outputs/metrics/gso_v3_rot5.json',
        10: 'outputs/metrics/gso_v3_rot10.json', 20: 'outputs/metrics/gso_v3_rot20.json'}
K, DELTA = [1, 2, 3], 3.0   # candidatas e limiar de descarte (dB)

for r in [0, 5, 10, 20]:
    A = jsonl(f'outputs/detA_rot{r}.jsonl')
    B = {k: jsonl(f'outputs/detB_rot{r}_drop{k}.jsonl') for k in K}
    keep = metr(KEEP[r])
    drop = {k: metr(f'outputs/metrics/drop_rot{r}_k{k}.json') for k in K}
    cenas = sorted(set(A) & set(keep) & set.intersection(*[set(B[k]) for k in K]))

    final, n_desc, n_certo = [], 0, 0
    for s in cenas:
        s4 = np.mean(A[s]['coarse'])                          # consistencia das 4 juntas
        sk = {k: np.mean(B[k][s]['coarse']) for k in K}       # consistencia de cada trio
        best = max(sk, key=sk.get)
        if sk[best] - s4 > DELTA:
            final.append(drop[best][s]); n_desc += 1; n_certo += (best == 3)
        else:
            final.append(keep[s])

    n = len(cenas)
    print(f'--- {r} graus ({n} cenas) ---')
    print(f'  manter as 4 vistas:      {np.mean([keep[s] for s in cenas]):.2f} dB')
    print(f'  descarte aleatorio:      {np.mean([np.mean([drop[k][s] for k in K]) for s in cenas]):.2f} dB')
    print(f'  descarte pelo detector:  {np.mean(final):.2f} dB   (descartou em {n_desc}/{n}, '
          f'acertou a vista 3 em {n_certo}/{max(n_desc, 1)})')
    print(f'  oraculo (tira a vista 3): {np.mean([drop[3][s] for s in cenas]):.2f} dB')
    print('\n--- margem: ganho do melhor trio sobre as 4 vistas (dB) ---')
for r in [0, 5, 10, 20]:
    A = jsonl(f'outputs/detA_rot{r}.jsonl')
    B = {k: jsonl(f'outputs/detB_rot{r}_drop{k}.jsonl') for k in K}
    cenas = sorted(set(A) & set.intersection(*[set(B[k]) for k in K]))
    g = [max(np.mean(B[k][s]['coarse']) for k in K) - np.mean(A[s]['coarse']) for s in cenas]
    print(f'  {r:>2} graus: min {np.min(g):6.2f}   mediana {np.median(g):6.2f}   max {np.max(g):6.2f}')