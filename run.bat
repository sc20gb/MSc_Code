@echo off
call conda activate CondaforDeepLearning
python src/fine_tune_with_gen.py --general_data_dir "D:\\Datasets" --save_dir "%cd%\\SavedModels"
echo Execution complete. Press any key to exit.
pause