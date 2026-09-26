import json, sys
import numpy as np

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    for tag in ['fine', 'coarse']:
        if tag not in rows[0]:
            continue
        P = np.array([r[tag] for r in rows])      # (n_cenas, 4 vistas)
        pred = 1 + P[:, 1:].argmin(1)             # candidatas 1..3 (vista 0 e referencia)
        print(f'{path} [{tag}]  acerto (achou a vista 3): {(pred == 3).mean():.2f}   '
              f'PSNR medio por vista: {np.round(P.mean(0), 2)}')