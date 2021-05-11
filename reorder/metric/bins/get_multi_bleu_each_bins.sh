MODEL=(
    "seg4"
    "seg6"
    "seg8"
    "seg10"
    "st"
    # "arbeam1"
    # "arbeam5"
    "arbeam10"
    "arnoload"
    # "arnomtl"
)
rm multi_bleu_results.txt
TOKENIZER=/media/xxx/Data/mosesdecoder/scripts/tokenizer/tokenizer.perl
for m in "${MODEL[@]}"; do
    bins=$(ls -l $m*.hyp | cut -d'_' -f3 | sort -g)
    for b in $bins; do
        hyp=${m}_bin_${b}_.hyp
        ref=${m}_bin_${b}_.ref

        cat $hyp | $TOKENIZER -q -l en -no-escape > $hyp.tok
        for v in "" "1" "2" "3"; do
            cat $ref$v | $TOKENIZER -q -l en -no-escape > $ref.tok$v
        done

        bleu=$(cat $hyp.tok \
        | perl multi-bleu.perl $ref.tok 2> /dev/null) \
        # | cut -d'=' -f2 | cut -d',' -f1)
        echo -e "$m\t$b\t$bleu" >> multi_bleu_results.txt
    done
done

# bleu=$(cat $hyp.tok \
#         | perl multi-bleu.perl $ref.tok 2> /dev/null \
#         | cut -d'=' -f2 | cut -d',' -f1)
