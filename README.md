Non-Autoregressive Speech Translation with CTC
===

Code for the paper "Investigating the Reordering Capability in CTC-based Non-Autoregressive End-to-End Speech Translation"  
In Findings of The Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021)  
by Shun-Po Chuang, Yung-Sung Chuang, Chih-Chiang Chang and Hung-yi Lee  

```
NARST_code/
├── README.md
├── espnet/
│   └── (...)
├── conf/
│   ├── decode_wo_asr_correct.yaml
│   ├── decode_wo_asr.yaml
│   ├── decode.yaml
│   ├── fbank.conf
│   ├── gpu.conf
│   ├── pitch.conf
│   ├── specaug.yaml
│   ├── train_encoder_st_correct_multitask_seg10.yaml
│   ├── train_encoder_st_correct_multitask_seg4.yaml
│   ├── train_encoder_st_correct_multitask_seg6.yaml
│   ├── train_encoder_st_correct_multitask_seg8.yaml
│   ├── train_encoder_st_correct.yaml
│   ├── train_encoder_st.yaml
│   ├── train_wo_asr.yaml
│   └── train.yaml
├── scripts/
│   ├── run_AR-MTL.sh
│   ├── run_AR.sh
│   ├── run_CTC-MTL-10.sh
│   ├── run_CTC-MTL-4.sh
│   ├── run_CTC-MTL-6.sh
│   ├── run_CTC-MTL-8.sh
│   └── run_CTC.sh
├── reorder/
│   ├── eval-align-kendalltau-text.ipynb
│   └── metric/
│       ├── simalign_files.py
│       ├── kendalltau_score.py
│       ├── get_data.py
│       ├── get_data.sh
│       ├── eval.sh
│       ├── align.sh
│       └── bins/
│           ├── multi-bleu.perl
│           └── get_multi_bleu_each_bins.sh
└── gradients/
    ├── gradients-ST.ipynb
    └── utils.py

```


## Setup
1. Download and install [ESPnet](https://github.com/espnet/espnet)
2. Replace `espnet/espnet/` with the `espnet/` folder.
3. Replace `espnet/egs/fisher_callhome_spanish/st1/conf/` with the `conf/` folder.
4. Copy `scripts/*.sh` to `espnet/egs/fisher_callhome_spanish/st1/`

## Training and Testing

We listed the train/test scripts to reproduce each results in Table 1 in our paper:

All scripts can be found in `scripts/`.

|       | Method                                    | Train/Test Script     | Load Pretrained Model\(Y/N\) |
|-------|-------------------------------------------|-----------------------|------------------------------|
| \(A\) | Autoregressive Model                      |                       |                              |
| \(a\) | Transformer \(b=10\)                      | run\_AR\.sh           | N                            |
| \(b\) | Transformer \+ MTL \(b=10\)               | run\_AR\-MTL\.sh      | N                            |
| \(c\) | Transformer \+ MTL \+ ASR init\. \(b=10\) | run\_AR\-MTL\.sh      | Y                            |
| \(d\) | Transformer \+ MTL \+ ASR init\. \(b=5\)  | run\_AR\-MTL\.sh      | Y                            |
| \(e\) | Transformer \+ MTL \+ ASR init\. \(b=1\)  | run\_AR\-MTL\.sh      | Y                            |
| \(B\) | Non\-Autoregressive Models \(Ours\)       |                       |                              |
| \(f\) | CTC                                       | run\_CTC\.sh          | Y                            |
| \(g\) | CTC \+ MTL at 4\-th layer                 | run\_CTC\-MTL\-4\.sh  | Y                            |
| \(h\) | CTC \+ MTL at 6\-th layer                 | run\_CTC\-MTL\-6\.sh  | Y                            |
| \(i\) | CTC \+ MTL at 8\-th layer                 | run\_CTC\-MTL\-8\.sh  | Y                            |
| \(j\) | CTC \+ MTL at 10\-th layer                | run\_CTC\-MTL\-10\.sh | Y                            |

## Evaluating reorder degree
The folder `reorder/` is used to measure $R_{acc}$ score and its relation with BLEU.

1. extract source, hypothesis and references from experiment folders 
```bash
bash get_data.sh
```
2. obtain alignments from SimAlign
```bash
bash align.sh
```
3. (optional) calculate $R_{acc}$ for individual file from an alignment
```bash
usage: kendalltau_score.py [-h] [--average] [--skip] [--acc] hyp_path [ref_path]

positional arguments:
  hyp_path    hypothesis alignments. Lines in the file should be indexed separated by TABs.
  ref_path    reference alignments. Same format as L1 file.

optional arguments:
  -h, --help  show this help message and exit
  --average   print file average instead of sentence scores.
  --skip      do not account for single source word sentences.
  --acc       report reorder accuracy by subtracting distance from 1 (1-d_tau).
```
4. calculate $R_{acc}$ for all model hypothesis and references
```bash
bash eval.sh
```

The notebook `eval-align-kendalltau-text.ipynb` contain code to plot the BLEU-$R_{acc}$ curve. After completing above 1-4, just run all cells in this notebook.

Other scripts (get_data.py, simalign_files.py, bins/multi-bleu.perl) are intermediate scripts.

## Gradient-based Visualization
The notebook `gradients-ST.ipynb`, which depends on `utils.py`, is used to calculate and visualize gradient norm in CTC models.


## Notes:

1. You need to specify the [Fisher-Callhome](https://catalog.ldc.upenn.edu/LDC2014T23) dataset directory in the training script. For example:
```clike
sfisher_speech=/groups/public/callhome_en_es/LDC2010S01/LDC2010S01
sfisher_transcripts=/groups/public/callhome_en_es/LDC2010T04/
split=local/splits/split_fisher

callhome_speech=/groups/public/callhome_en_es/LDC96S35
callhome_transcripts=/groups/public/callhome_en_es/LDC96T17
split_callhome=local/splits/split_callhome
```
Download them from:
- 1) Fisher Spanish Speech https://catalog.ldc.upenn.edu/LDC2010S01
- 2) CALLHOME Spanish Speech https://catalog.ldc.upenn.edu/LDC96S35
- 3) Fisher and CALLHOME Spanish--English Speech Translation https://catalog.ldc.upenn.edu/LDC2014T23
2. You need to specify `asr_model=${YOUR_ASR_MODEL_PATH}` in the training script if `Load Pretrained Model(Y/N)=Y`
3. Set `stage=0` if you need to start from data preparation. Otherwise set it to `4` for training, `5` for inference
4. When inference the AR models, set `beam-size` in `decode.yaml` and `decode_wo_asr.yaml` to the beam search size you want. (beam size=10,5,1 in our experiments)

### Knowledge Distillation

We use sequence-level knowledge distillation for our CTC-models training. Just simply use a pretrained MT model like this:
https://github.com/espnet/espnet/blob/master/egs/fisher_callhome_spanish/mt1/RESULTS.md
to produce the hypothesis for the whole training set. 

And then replace the ground truth sentences in `espnet/egs/fisher_callhome_spanish/st1/dump/train_sp.en/deltafalse/data_bpe8000.lc.rm.json` with decoded sentences.

(the directory `espnet/egs/fisher_callhome_spanish/st1/dump/train_sp.en/deltafalse/` with be created after running the preprocessing stage 0)

We provide our decode results here:
1) MT: https://www.dropbox.com/s/153cujsyghn3obn/data_bpe8000.lc.rm.json.kdmt.tar.gz
2) ST: https://www.dropbox.com/s/owwz2r4o4dbibnn/data_bpe8000.lc.rm.json.kdst.tar.gz

Both of them are produced by the models trained by ourselves. The beam search size for decoding is 1. 
You may need to replace all dummy path with `DUMMY` to your real path to the feats data.

