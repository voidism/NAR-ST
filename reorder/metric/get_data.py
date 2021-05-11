import sys
from tqdm.auto import tqdm
import json
import argparse
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sacremoses import MosesTokenizer, MosesDetokenizer


if __name__ == "__main__":
    def spacetok(x):
        return x.strip().split(' ')
    mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')
    parser = argparse.ArgumentParser()
    parser.add_argument("decode_root", type=str, help="")
    parser.add_argument("data_root", type=str, help="")
    parser.add_argument("--outdir", type=str, help="")
    parser.add_argument("--tag", type=str, help="")
    parser.add_argument("--allrefs", action="store_true", help="")

    
    args = parser.parse_args()


    decode_root = Path(args.decode_root)
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    VER = ("", "1", "2", "3") if args.allrefs else ("",)

    # read hyp and ref
    hyp_name = decode_root / "hyp.wrd.trn.detok.lc.rm"

    with open(hyp_name, "r") as f:
        hyps = f.readlines()
    # with open(root + "/data.json", "r") as f:
    #     test_json = json.load(f)["utts"]

    # read src
    with open(data_root / "data_bpe1000.lc.rm.json", "r") as f:
        test_json = json.load(f)
        test_json = test_json["utts"]

    srcs = []
    for i, (key, info) in enumerate(test_json.items()):
        s = info["output"][1]["text"].strip()
        s = md.detokenize(spacetok(s))
        srcs.append(s)
    hyps = [ t.strip() for t in hyps ]
    # print("srcs", len(srcs))
    # print("refs", len(refs))
    # print("hyps", len(hyps))
    assert len(srcs) == len(hyps)

    # read refs
    refs = []
    for v in VER:
        ref_name = decode_root / f"ref{v}.wrd.trn.detok.lc.rm" 
        with open(ref_name, "r") as f:
            rf = f.readlines()
        refs.append([ t.strip() for t in rf ])
        assert len(srcs) == len(rf)

    with open(outdir / f"{args.tag}.src", "w") as f:
        for t in srcs:
            f.write(t+"\n")
    with open(outdir / f"{args.tag}.hyp", "w") as f:
        for t in hyps:
            f.write(t+"\n")
    for i,v in enumerate(VER):
        with open(outdir / f"{args.tag}.ref{v}", "w") as f:
            for t in refs[i]:
                f.write(t+"\n")