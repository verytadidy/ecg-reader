python inference_and_visualization.py \
    --checkpoint /Users/zebra/Documents/kaggle/ecg_project/experiments/run_20251116_173214/checkpoints/best.pth \
    --image /Volumes/movie/work/physionet-ecg-image-digitization/train/7663343/7663343-0001.png \
    --output_dir ./inference_results


python inference_and_visualization.py \
    --checkpoint /Users/zebra/Documents/kaggle/ecg_project/experiments/run_20251116_173214/checkpoints/best.pth \
    --image /Volumes/movie/work/physionet-ecg-image-digitization/train/7663343/7663343-0001.png \
    --gt_csv /Volumes/movie/work/physionet-ecg-image-digitization/train/7663343/7663343.csv \
    --output_dir ./inference_results \
    --show_intermediate

python analyze_constraints.py \
    --train_dir /Volumes/movie/work/physionet-ecg-image-digitization/train \
    --train_csv /Volumes/movie/work/physionet-ecg-image-digitization/train.csv \
    --output constraint_analysis.json \
    --plot constraint_analysis.png