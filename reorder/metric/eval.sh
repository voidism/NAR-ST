DATA=(
    "fisher_dev.en"
    "fisher_dev2.en"
    "fisher_test.en"
)
MODEL=(
    "arbeam1"
    "arbeam5"
    "arbeam10"
    "arnomtl"
    "arnoload"
    "st"
    "seg4"
    "seg6"
    "seg8"
    "seg10"
    # "muse"
    # "moses"
    "rand"
)
OUT=kendalltau
mkdir -p $OUT
AVG="" # "--average"

# order only
echo "-- scoring order --"
for d in "${DATA[@]}"; do
    echo "scoring $d refs"
    for v in "" "1" "2" "3"; do
        m=${MODEL[0]}
        python kendalltau_score.py $AVG \
        alignments/$d.st.ref$v.itermax \
        | tee $OUT/$d.ref$v
    done
    echo "scoring $d models"
    for m in "${MODEL[@]}"; do
        if [[ $m == "rand" ]];  then
            continue
        fi
        mkdir -p $OUT
        python kendalltau_score.py $AVG \
        alignments/$d.$m.hyp.itermax \
        | tee $OUT/$d.$m
    done
done

# correctness
echo "-- scoring correctness --"
for d in "${DATA[@]}"; do
    echo "## scoring $d"
    for v in "" "1" "2" "3"; do
        echo "scoring ref$v"
        m=${MODEL[0]}
        python kendalltau_score.py --acc $AVG \
        alignments/$d.st.ref$v.itermax \
        alignments/$d.st.ref$v.itermax \
        | tee $OUT/$d.ref$v.acc

        for m in "${MODEL[@]}"; do
            # echo "scoring $d $m (ref$v)"
            mkdir -p $OUT

            if [[ $m == "rand" ]];  then
                python kendalltau_score.py --acc $AVG \
                alignments/$d.st.ref$v.itermax \
                --randperm \
                | tee $OUT/$d.rand.$v.acc
            else
                python kendalltau_score.py --acc $AVG \
                alignments/$d.$m.hyp.itermax \
                alignments/$d.st.ref$v.itermax \
                | tee $OUT/$d.$m.$v.acc
            fi
        done
    done
done
