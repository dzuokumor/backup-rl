@echo off
set PYTHON=C:\Users\user\PycharmProjects\LoRA\lora-venv\Scripts\python.exe
cd /d C:\Users\user\PycharmProjects\RL-for-PGM
echo evaluating all agents (baselines + best trained models, 50 episodes each)...
%PYTHON% evaluate.py --all --baselines --episodes 50
pause
