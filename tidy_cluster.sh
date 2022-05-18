# remove temporary files from the cluster

cd ~/

# remove the output files eg ArrayTrainDQN_13.o4647097.2
rm *.o*.*

# remove any model files
rm -r ~/mymujoco/rl/models/*

# remove any wandb files
rm -r ~/mymujoco/rl/wandb