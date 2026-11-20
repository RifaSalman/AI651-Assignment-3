"""Question 1 helpers: unique message encoding + exhaustive check for K=8."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)


def phi_values(K: int):
    """Base-(K+1) place encoding so the sum uniquely recovers (n_neg, n_zero, n_pos)."""
    B = K + 1
    # phi(-1), phi(0), phi(+1)
    return 1.0, float(B), float(B * B)


def decode_counts(s: float, K: int):
    B = K + 1
    s_int = int(round(s))
    n_neg = s_int % B
    n_zero = (s_int // B) % B
    n_pos = s_int // (B * B)
    return n_neg, n_zero, n_pos


def verify_all_triples(K: int = 8):
    phi_m1, phi_0, phi_p1 = phi_values(K)
    ok = 0
    total = 0
    # every (n_neg, n_zero, n_pos) with 1 <= n_neg+n_zero+n_pos <= K
    for n_neg in range(0, K + 1):
        for n_zero in range(0, K + 1 - n_neg):
            for n_pos in range(0, K + 1 - n_neg - n_zero):
                n = n_neg + n_zero + n_pos
                if n < 1:
                    continue
                total += 1
                s = n_neg * phi_m1 + n_zero * phi_0 + n_pos * phi_p1
                rec = decode_counts(s, K)
                if rec == (n_neg, n_zero, n_pos):
                    ok += 1
                else:
                    return {
                        "ok": False,
                        "failed": {
                            "true": [n_neg, n_zero, n_pos],
                            "sum": s,
                            "decoded": list(rec),
                        },
                    }
    return {
        "ok": True,
        "K": K,
        "phi_minus1": phi_m1,
        "phi_0": phi_0,
        "phi_plus1": phi_p1,
        "base": K + 1,
        "checked_triples": total,
        "passed": ok,
    }


def nonsteady_message_demo(rng: np.random.Generator):
    """phi(x)=1 for |x|=1 and 0 for x=0: sum = count of non-steady neighbours."""
    phi = {-1: 1.0, 0: 0.0, 1: 1.0}
    examples = []
    for _ in range(5):
        n = int(rng.integers(1, 9))
        trends = rng.choice([-1, 0, 1], size=n)
        s = float(sum(phi[int(x)] for x in trends))
        true = int(np.sum(np.abs(trends) == 1))
        examples.append({"trends": trends.tolist(), "sum": s, "nonsteady_count": true})
    return examples


def affine_collision_example():
    """Two neighbourhoods with same n and S but different # of steady sensors."""
    # N1: {-1, 0, +1}  -> n=3, S=0, steady=1
    # N2: {0, 0, 0}    -> n=3, S=0, steady=3
    n1 = np.array([-1, 0, 1])
    n2 = np.array([0, 0, 0])
    return {
        "N1": n1.tolist(),
        "N2": n2.tolist(),
        "n": int(len(n1)),
        "S": [int(n1.sum()), int(n2.sum())],
        "steady_counts": [int((n1 == 0).sum()), int((n2 == 0).sum())],
        "note": "affine phi(x)=wx+c => sum = wS + c*n, identical for both",
    }


def aggregation_table_examples():
    """Concrete collision pairs used in the Q1(c) 'No' answers."""
    # Same mean vector, different |N|:
    # A: two slowing -> mean (1,0,0); B: one slowing -> mean (1,0,0)
    # Same max vector: any set that includes at least one of each vs one of each only once
    return {
        "mean_collision_half_slowing": {
            "A": [-1, -1],  # 100% slowing, mean=(1,0,0), half? yes (2/2)
            "B": [-1],  # 100% slowing, mean=(1,0,0), half? yes — bad example
            "better": {
                "A": [-1, -1, 0, 0],  # half slowing (2/4), sum=(2,2,0), mean=(0.5,0.5,0)
                "B": [-1, 0],  # half slowing (1/2), mean=(0.5,0.5,0)
                "note": "same mean, same answer — need different answers",
            },
            "use": {
                "A": [-1, -1, 1],  # 2/3 slowing > half, mean=(2/3,0,1/3)
                "B": [-1, 1],  # 1/2 slowing not > half, mean=(0.5,0,0.5)
                "comment": "different means — for SUM collision:",
            },
            "sum_collision_count_steady": {
                "A": [0, 0, -1],  # steady=2, sum=(1,2,0)
                "B": [0, 0],  # steady=2, sum=(0,2,0) different
            },
            "sum_ok_steady": "sum[:,1] = count steady",
            "mean_collision_steady_count": {
                "A": [0, 0],  # mean=(0,1,0), steady count 2
                "B": [0],  # mean=(0,1,0), steady count 1
                "agg": [0, 1, 0],
                "answers": [2, 1],
            },
            "max_collision_steady_count": {
                "A": [0, 0, -1],  # max=(1,1,0), steady count 2
                "B": [0, -1],  # max=(1,1,0), steady count 1
                "agg": [1, 1, 0],
                "answers": [2, 1],
            },
            "sum_collision_more_than_half_slowing": {
                "A": [-1, -1, 1],  # 2/3 > 1/2, sum=(2,0,1)
                "B": [-1, -1, 1, 1, 0],  # wait need same sum
                "pair": {
                    "A": [-1, -1],  # 2/2 > half, sum=(2,0,0)
                    "B": [-1, -1, 0, 0],  # 2/4 = half NOT >, sum=(2,2,0) different sums
                },
                "true_collision_for_sum": None,  # sum actually CAN do this with n from sum of coords
            },
        }
    }


def main():
    rng = np.random.default_rng(7)
    result = {
        "affine_collision": affine_collision_example(),
        "encoding_verify_K8": verify_all_triples(8),
        "nonsteady_examples": nonsteady_message_demo(rng),
        "phi_nonsteady": {"-1": 1.0, "0": 0.0, "1": 1.0},
    }
    path = OUT / "q1_verify.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["encoding_verify_K8"], indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
