@echo off
set PYTHON=C:\Users\user\PycharmProjects\LoRA\lora-venv\Scripts\python.exe
cd /d C:\Users\user\PycharmProjects\RL-for-PGM
echo ppo optimized training (1m steps, reduced faults, auto-recovery, survival bonus)...
%PYTHON% training/train_ppo_optimized.py %*
pause
