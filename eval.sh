
for alg in "./logs/*.pth"
do
    for pth in $alg
    do
        python3 eval.py $pth
    done
done