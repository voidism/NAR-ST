from scipy import stats
import argparse
import numpy as np
import pdb
import math
from typing import NamedTuple, List
Align = NamedTuple(
    "Align",
    [
        ("hyplen", int),  
        ("result", List[int]),  
    ],
)

def kendalltauscore(aln1, aln2):
    tau, p = stats.kendalltau(aln1, aln2, variant='c') # 'c' for different possible values in alignment
    if tau != tau:
        dist = 0.0
    else:
        dist = (1-tau)/2 #  Z * (1-tau)/2 / Z  ( https://quantdare.com/when-distance-is-the-issue/ )
    return dist

def _extract(l, skip):
    sid, srclen, hyplen, a = l.strip().split("\t") # 0-0 1-1 2-3 ...
    srclen, hyplen = int(srclen), int(hyplen)
    if srclen == 1 and skip:
        return None
    a = [ tuple(map(int, x.split('-'))) for x in a.split()] # [(0,0) (1,1) (2,3) ... ]
    # align.sh is (src-tgt)
    a = dict(a)
    accessed = np.zeros(hyplen, dtype=np.long)
    alignment = []
    for i in range(srclen):
        if i in a:
            p = a[i]
        elif i==0:
            p = 0
        else:
            p = alignment[i-1][0]
        alignment.append((p, accessed[p]))
        accessed[p] += 1
    
    accessed -= 1
    offset = np.cumsum(accessed) - accessed
    result = [ p+ac+offset[p] for p, ac in alignment]
    return Align(hyplen=hyplen, result=result)

def bp(hyplen, reflen):
    t, r = hyplen, reflen
    return 1 if t > r else math.exp(1-r/t)

def dummy(h, rand=False):
    if rand:
        dum = np.random.permutation(h.result) #sorted(h.result)
    else:
        dum = sorted(h.result)
    return Align(h.hyplen, dum)


def score_files(hyp_path, ref_path, skip=False, average=False, accuracy=False, randperm=False):
    hyp_aln = []
    with open(hyp_path, "r") as f:
        for line in f:
            hyp_aln.append(_extract(line, skip=skip))

    ref_aln = []
    if ref_path != None:
        with open(ref_path, "r") as f:
            for line in f:
                ref_aln.append(_extract(line, skip=skip))
    else:
        ref_aln = [ None if h is None else dummy(h, rand=randperm) for h in hyp_aln ]
    
    scores = []
    for h, r in zip(hyp_aln, ref_aln):        
        if r==None: # skipped
            if average:
                continue
            else:
                s = 0
        else:
            s = kendalltauscore(h.result,r.result)
        if accuracy:
            s = (1 - s**0.5)*bp(h.hyplen,r.hyplen)

        scores.append(s)
        if not average:
            print(f"{s:.04f}")

    if average:
        avg = 0 if len(scores)==0 else (sum(scores) / len(scores))
        print(f"{avg*100:.02f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp_path", type=str, help="hypothesis alignments. Lines in the file should be indexed separated by TABs.")
    parser.add_argument("ref_path", type=str, nargs="?", const=None, help="reference alignments. Same format as L1 file.")    
    parser.add_argument("--average", action="store_true", help="print file average instead of sentence scores.")  
    parser.add_argument("--skip", action="store_true", help="do not account for single source word sentences.")
    parser.add_argument("--acc", dest="accuracy", action="store_true", help="report reorder accuracy by this: (1-d_tau**0.5)*BP.")
    parser.add_argument("--randperm", action="store_true", help="use random permutation as dummy alignment instead of monotonic.")
    
    args = parser.parse_args()

    score_files(**vars(args))
