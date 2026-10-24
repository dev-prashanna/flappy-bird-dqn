This directory preserves the pre-PPO working tree: the Double DQN implementation
and the artifacts it produced (`best_model.pth`, `final_model.pth`,
`training_log.txt`). Run these scripts with `legacy/` as the working directory
if investigating the old implementation; they load and save the weights in this
directory. They use a different environment API, state dimension, physics,
reward, and algorithm, so their results are not comparable to the PPO
experiments at the repository root, and no current command reads or overwrites
these files.
