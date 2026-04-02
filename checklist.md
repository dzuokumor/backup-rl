# rubric checklist

## environment
- [x] custom environment based on capstone/mission (Lagos/EKEDC power grid)
- [x] action space defined (53 discrete actions)
- [x] observation space defined (117-dim continuous)
- [x] reward structure defined and implemented
- [x] start state defined
- [x] terminal conditions defined
- [x] visualization using OpenGL + pygame (3D nighttime Lagos city)
- [x] static file showing random agent (main.py --random --no-render, logs to CSV)
- [ ] agent diagram in report (mermaid code is in report.md, needs to be rendered as image)

## training
- [x] DQN implemented and trained (SB3, 10 runs x 100k + 1M best)
- [x] REINFORCE implemented and trained (custom PyTorch, 10 runs x 100k + 1M best)
- [x] PPO implemented and trained (SB3, 10 runs x 100k + 1M best)
- [x] A2C implemented and trained (SB3, 10 runs x 100k + 1M best)
- [x] all four use same environment
- [x] hyperparameter tuning with 10 runs each (40 total)
- [x] hyperparameter tables in report with results

## video
- [ ] screen share with camera on
- [ ] state the problem
- [ ] state agent behaviour
- [ ] explain reward structure
- [ ] state objective of the agent
- [ ] run simulation with best agent (PPO), show GUI and terminal output
- [ ] explain agent performance

## report
- [x] project overview (done)
- [x] environment description: agent, action space, observation space, reward (done)
- [x] system analysis and design: DQN, REINFORCE, PPO, A2C descriptions (done)
- [x] implementation: hyperparameter tables with 10 rows each (done)
- [x] results: cumulative rewards plot (done)
- [x] results: training stability plots - DQN loss, PG entropy (done)
- [x] results: episodes to converge plot (done)
- [x] results: generalization test (done)
- [x] conclusion and discussion (done)
- [ ] render mermaid diagram as image and insert
- [ ] 7-10 pages (check after formatting)
- [ ] saved as PDF
- [x] report written in student voice (humanized)

## repo
- [x] requirements.txt
- [x] README.md
- [x] environment/custom_env.py
- [x] environment/rendering.py
- [x] training/dqn_training.py
- [x] training/pg_training.py
- [x] models/dqn/ (all models saved)
- [x] models/pg/ (all models saved)
- [x] main.py entry point
- [ ] repo named "student_name_rl_summative" (currently "backup-rl" and "RL-for-PGM")

## still needs doing
1. render mermaid diagram to image (can use mermaid.live or VS Code preview)
2. record video demo
3. format report as PDF (insert plot images, diagram)
4. rename/create final repo as "student_name_rl_summative"
5. submit PDF to Canvas
