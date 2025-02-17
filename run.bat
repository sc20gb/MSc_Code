@echo off
call conda activate CondaforDeepLearning
python src/fine_tune_with_gen.py
echo Execution complete. Press any key to exit.
pause