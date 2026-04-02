@echo off
setlocal

set PYTHON=C:\Users\user\PycharmProjects\LoRA\lora-venv\Scripts\python.exe
set PROJECT=C:\Users\user\PycharmProjects\RL-for-PGM

cd /d %PROJECT%

echo training dqn (10 runs)...
%PYTHON% training/dqn_training.py %*
if errorlevel 1 (
    echo dqn training interrupted or failed
    echo run again with --resume to continue
    pause
    exit /b 1
)

echo training ppo (10 runs)...
%PYTHON% training/pg_training.py --algorithm ppo %*
if errorlevel 1 (
    echo ppo training interrupted or failed
    echo run again with --resume to continue
    pause
    exit /b 1
)

echo training a2c (10 runs)...
%PYTHON% training/pg_training.py --algorithm a2c %*
if errorlevel 1 (
    echo a2c training interrupted or failed
    echo run again with --resume to continue
    pause
    exit /b 1
)

echo training reinforce (10 runs)...
%PYTHON% training/pg_training.py --algorithm reinforce %*
if errorlevel 1 (
    echo reinforce training interrupted or failed
    echo run again with --resume to continue
    pause
    exit /b 1
)

echo all training complete
echo generating plots...
%PYTHON% results/generate_plots.py

echo done
pause
