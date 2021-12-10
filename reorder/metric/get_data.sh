DATA=(
    "fisher_dev.en"
    "fisher_dev2.en"
    "fisher_test.en"
)
OUT=dump
mkdir -p $OUT

espnet_dir="/home/xxx/Projects/NAR-ST/espnet"
for d in "${DATA[@]}"; do
    data_dir="${espnet_dir}/egs/fisher_callhome_spanish/st1/dump/${d}/deltafalse/"

    t="arnoload"
    echo ${d}.${t}
    decode_dir="${espnet_dir}/egs/fisher_callhome_spanish/st1/exp/train_sp.en_lc.rm_pytorch_train_wo_asr_bpe1000/decode_${d}_decode_wo_asr"
    python get_data.py --allrefs $decode_dir $data_dir --outdir $OUT --tag "${d}.${t}"

    t="arnomtl"
    echo ${d}.${t}
    decode_dir="${espnet_dir}/egs/fisher_callhome_spanish/st1/exp/train_sp.en_lc.rm_pytorch_train_bpe1000/decode_${d}_decode"
    python get_data.py --allrefs $decode_dir $data_dir --outdir $OUT --tag "${d}.${t}"

    for b in 1 5; do
        t="arbeam${b}"
        echo ${d}.${t}
        decode_dir="${espnet_dir}/egs/fisher_callhome_spanish/st1/exp/train_sp.en_lc.rm_pytorch_at_w_asr_w_pretrain/prev_decode_b${b}/decode_${d}_decode-b${b}"
        python get_data.py --allrefs $decode_dir $data_dir --outdir $OUT --tag "${d}.${t}"
    done

    t="arbeam10"
    echo ${d}.${t}
    decode_dir="${espnet_dir}/egs/fisher_callhome_spanish/st1/exp/train_sp.en_lc.rm_pytorch_at_w_asr_w_pretrain/prev_decode_b10/decode_${d}_decode"
    python get_data.py --allrefs $decode_dir $data_dir --outdir $OUT --tag "${d}.${t}"
done

for d in "${DATA[@]}"; do
    data_dir="${espnet_dir}/egs/fisher_callhome_spanish/st1/dump/${d}/deltafalse/"

    t="st"
    echo ${d}.${t}
    decode_dir="${espnet_dir}/egs/fisher_callhome_spanish/st1/exp/train_sp.en_lc.rm_pytorch_encoder_st_init_w_asr_correct_bpe8k_specaug_kdmt/decode_${d}_decode_wo_asr_correct"
    python get_data.py --allrefs $decode_dir $data_dir --outdir $OUT --tag "${d}.${t}"

    for i in 4 6 8 10; do        
        t="seg${i}"
        echo ${d}.${t}
        decode_dir="${espnet_dir}/egs/fisher_callhome_spanish/st1/exp/train_sp.en_lc.rm_pytorch_encoder_st_init_w_asr_correct_multitask_seg${i}_bpe8k_specaug_kdmt/decode_${d}_decode_wo_asr_correct"
        python get_data.py $decode_dir $data_dir --outdir $OUT --tag "${d}.${t}"
    done
done


#     exp/
# train_sp.en_lc.rm_pytorch_encoder_st_init_w_asr_correct_multitask_seg8_bpe8k_specaug_kdmt
# train_sp.en_lc.rm_pytorch_encoder_st_init_w_asr_correct_bpe8k_specaug_kdmt
# train_sp.en_lc.rm_pytorch_encoder_st_init_w_asr_correct_multitask_seg10_bpe8k_specaug_kdmt
# train_sp.en_lc.rm_pytorch_encoder_st_init_w_asr_correct_multitask_seg6_bpe8k_specaug_kdmt
# train_sp.en_lc.rm_pytorch_encoder_st_init_w_asr_correct_multitask_seg4_bpe8k_specaug_kdmt


    # t="moses"
    # echo ${d}.${t}
    # # decode_dir="${espnet_dir}/egs/fisher_callhome_spanish/st1/moses"
    # cp moses_results/${d%.en}.${t}.txt ${OUT}/${d}.${t}.hyp

    # t="moses.nolm"
    # echo ${d}.${t}
    # cp moses_results/${d%.en}.${t}.txt ${OUT}/${d}.${t}.hyp
    
    # t="muse"
    # echo ${d}.${t}
    # decode_dir="${espnet_dir}/egs/fisher_callhome_spanish/st1/muse"
    # cut -f2 $decode_dir/w2w.${d}.txt > ${OUT}/${d}.${t}.hyp
   
    # t="muse.cpy"
    # echo ${d}.${t}
    # decode_dir="${espnet_dir}/egs/fisher_callhome_spanish/st1/muse"
    # cut -f2 $decode_dir/w2w_copy.${d}.txt > ${OUT}/${d}.${t}.hyp
