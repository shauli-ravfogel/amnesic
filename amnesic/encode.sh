for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do

    python3 encode.py \
        --model bert-base-uncased \
        --labels upos \
        --layer ${layer}
done