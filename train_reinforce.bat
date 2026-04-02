@echo off
set PYTHON=C:\Users\user\PycharmProjects\LoRA\lora-venv\Scripts\python.exe
cd /d C:\Users\user\PycharmProjects\RL-for-PGM
echo training reinforce (10 runs)...
%PYTHON% training/pg_training.py --algorithm reinforce %*
pause
