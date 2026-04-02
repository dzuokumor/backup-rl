@echo off
set PYTHON=C:\Users\user\PycharmProjects\LoRA\lora-venv\Scripts\python.exe
cd /d C:\Users\user\PycharmProjects\RL-for-PGM
echo training a2c (10 runs)...
%PYTHON% training/pg_training.py --algorithm a2c %*
pause
