#!/bin/bash

dataset_path="datasets/euroc/"

datasets=(
    MH_01_easy
    MH_02_easy
    MH_03_medium
    MH_04_difficult
    MH_05_difficult
    V1_01_easy
    V1_02_medium
    V1_03_difficult
    V2_01_easy
    V2_02_medium
    V2_03_difficult
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
            python main.py --dataset $dataset_name --no-viz --save-as euroc/no_calib/$dataset --config config/eval_no_calib.yaml
        else
            python main.py --dataset $dataset_name --no-viz --save-as euroc/calib/$dataset --config config/eval_calib.yaml
        fi
    done
fi

for dataset in ${datasets[@]}; do
    dataset_name="$dataset_path""$dataset"/
    echo ${dataset_name}
    if [ "$no_calib" = true ]; then
        evo_ape tum groundtruths/euroc/$dataset.txt logs/euroc/no_calib/$dataset/$dataset.txt -as
    else
        evo_ape tum groundtruths/euroc/$dataset.txt logs/euroc/calib/$dataset/$dataset.txt -as
    fi

done
