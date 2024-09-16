STEP=50
echo results/checkpoint-0
python evaluation/eval_mmlu.py -s results/checkpoint-0

for i in {1..4}
do
    echo results/checkpoint-$((STEP * i))
    python evaluation/eval_mmlu.py -s results/checkpoint-$((STEP * i)) -m output/alpaca-gpt4_20000_fedavg_c20s2_i10_b16a1_l512_r32a64_20240828083620/checkpoint-$((STEP * i)) 
done

