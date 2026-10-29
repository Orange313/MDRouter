"""Linear-gain nDCG@10 and the graded B1-L upper bound."""
import math

def discount(rank, cutoff=10):
    return 1.0 / math.log2(rank + 1) if 1 <= rank <= cutoff else 0.0

def ndcg(ranking, qrels, cutoff=10):
    if not qrels or max(qrels.values(), default=0) <= 0:
        raise ValueError('Positive qrels are required; missing labels are not zero quality')
    if len(set(ranking)) != len(ranking):
        raise ValueError('Duplicate page IDs in a ranking')
    ideal = sum(g * discount(i, cutoff) for i, g in enumerate(sorted(qrels.values(), reverse=True)[:cutoff], 1))
    return sum(qrels.get(p, 0) * discount(i, cutoff) for i, p in enumerate(ranking[:cutoff], 1)) / ideal

def b1l(reference, alternative, reference_top10_qrels, cutoff=10):
    # Only judgments for pages actually returned in the reference top ten are used.
    labels = {p: reference_top10_qrels.get(p, 0) for p in reference[:cutoff]}
    ideal_in = sum(g * discount(i, cutoff) for i, g in enumerate(sorted(labels.values(), reverse=True), 1))
    if ideal_in == 0:
        return 0.0
    positions = {p: i for i, p in enumerate(alternative[:cutoff], 1)}
    numerator = sum(labels[p] * max(0.0, discount(i, cutoff) - discount(positions.get(p, cutoff + 1), cutoff))
                    for i, p in enumerate(reference[:cutoff], 1))
    return min(1.0, numerator / ideal_in)

def stable_ranking(ids, scores):
    if len(ids) != len(scores) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError('Invalid scores')
    return sorted(range(len(ids)), key=lambda i: (-float(scores[i]), int(ids[i])))
