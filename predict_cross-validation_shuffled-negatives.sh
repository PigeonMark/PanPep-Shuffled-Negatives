for CV in 0 1 2 3 4; do
	python PanPep.py --learning_setting zero-shot --input ./Data/cross-validation_shuffled-negatives_splits/test_fold_$CV.csv --output ./Output/cross-validation_shuffled-negatives_splits/panpep_predictions_$CV.csv
done
