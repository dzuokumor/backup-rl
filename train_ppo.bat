@echo off
set PYTHON=C:\Users\user\PycharmProjects\LoRA\lora-venv\Scripts\python.exe
cd /d C:\Users\user\PycharmProjects\RL-for-PGM
echo training ppo (10 runs)...
%PYTHON% training/pg_training.py --algorithm ppo %*
pause
