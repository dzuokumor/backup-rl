@echo off
setlocal

set PYTHON=C:\Users\user\PycharmProjects\LoRA\lora-venv\Scripts\python.exe
set PROJECT=C:\Users\user\PycharmProjects\RL-for-PGM

cd /d %PROJECT%

echo resuming all training from checkpoints...
echo completed runs will be skipped automatically

echo resuming dqn...
%PYTHON% training/dqn_training.py --resume
if errorlevel 1 (
    echo dqn interrupted, run this bat again to continue
    pause
    exit /b 1
)

echo resuming ppo...
%PYTHON% training/pg_training.py --algorithm ppo --resume
if errorlevel 1 (
    echo ppo interrupted, run this bat again to continue
    pause
    exit /b 1
)

echo resuming a2c...
%PYTHON% training/pg_training.py --algorithm a2c --resume
if errorlevel 1 (
    echo a2c interrupted, run this bat again to continue
    pause
    exit /b 1
)

echo resuming reinforce...
%PYTHON% training/pg_training.py --algorithm reinforce --resume
if errorlevel 1 (
    echo reinforce interrupted, run this bat again to continue
    pause
    exit /b 1
)

echo all training complete
echo generating plots...
%PYTHON% results/generate_plots.py

echo done
pause
