DATA=(
    "fisher_dev.en"
    "fisher_dev2.en"
    "fisher_test.en"
)
MODEL=(
    "arbeam1"
    "arbeam5"
    "arbeam10"
    "arnoload"
    "arnomtl"
    "st"
    "seg4"
    "seg6"
    "seg8"
    "seg10"
    # "muse"
    # "moses"
)
OUT=alignments
mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=1
for d in "${DATA[@]}"; do
    for m in "${MODEL[@]}"; do
        src=$d.st.src

        cands=(
            "$d.$m.hyp"
            "$d.st.ref"
            "$d.st.ref1"
            "$d.st.ref2"
            "$d.st.ref3"
        )
        for f in "${cands[@]}"; do
            o=$OUT/$f.itermax
            if [ -f $o ]; then
                echo "%%%%%%%%%% $o exists, skipping alignment. %%%%%%%%%%"
            else    
                echo "%%%%%%%%%% aligning files $src $f ... %%%%%%%%%%"
                rm -f L1.tmp L2.tmp # just to be sure
                cat -n dump/$src > L1.tmp
                cat -n dump/$f > L2.tmp
                python simalign_files.py L1.tmp L2.tmp -model xlmr --matching-methods i -device cuda -output ${o%.itermax}
                echo "%%%%%%%%%% done. alignment written to $o %%%%%%%%%%"
            fi
        done
    done    

done

# --token-type word 