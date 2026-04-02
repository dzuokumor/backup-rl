@echo off
set PYTHON=C:\Users\user\PycharmProjects\LoRA\lora-venv\Scripts\python.exe
cd /d C:\Users\user\PycharmProjects\RL-for-PGM
echo 1m step training (best hyperparams per algorithm)...
%PYTHON% training/train_1m.py %*
pause
