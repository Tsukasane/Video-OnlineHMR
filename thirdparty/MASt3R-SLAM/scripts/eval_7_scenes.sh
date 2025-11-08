#!/bin/bash
dataset_path="datasets/7-scenes/"
datasets=(
    chess
    fire
    heads
    office
    pumpkin
    redkitchen
    stairs
)

no_calib=false
print_only=false
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-calib)
            no_calib=true
            ;;
        --print)
            print_only=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

if [ "$print_only" = false ]; then
    for dataset in ${datasets[@]}; do
        dataset_name="$dataset_path""$dataset"/
        if [ "$no_calib" = true ]; then
            python main.py --dataset $dataset_name --no-viz --save-as 7-scenes/no_calib/$dataset --config config/eval_no_calib.yaml
        else
            python main.py --dataset $dataset_name --no-viz --save-as 7-scenes/calib/$dataset --config config/eval_calib.yaml
        fi
    done
fi

for dataset in ${datasets[@]}; do
    dataset_name="$dataset_path""$dataset"/
    echo ${dataset_name}
    if [ "$no_calib" = true ]; then
        evo_ape tum groundtruths/7-scenes/$dataset.txt logs/7-scenes/no_calib/$dataset/$dataset.txt -as
    else
        evo_ape tum groundtruths/7-scenes/$dataset.txt logs/7-scenes/calib/$dataset/$dataset.txt -as
    fi

done
