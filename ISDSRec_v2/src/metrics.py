import numpy as np

def hr_ndcg_at_k(ranked_items, target, k=10):
    top=list(ranked_items[:k])
    if target not in top: return 0.0,0.0
    rank=top.index(target)+1
    return 1.0, 1.0/np.log2(rank+1.0)

def aggregate(rows, keys=('hr','ndcg')):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
