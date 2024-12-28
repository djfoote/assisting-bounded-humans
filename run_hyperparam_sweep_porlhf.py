from multiprocessing import Process

import numpy as np

# import porlhf_paper_experiment as experiment
import porlhf_paper_experiment_logging_per_epoch as experiment
import wandb

# Configure experiment

PROJECT_NAME = "porlhf_hyperparam_sweep"

TOTAL_QUERIES = 28
NUM_EPOCHS = 20000
NUM_SEEDS = 1

MAX_REWARD_PENALTY = 10
HPARAM_GRID_SIZE = 11

train_hyperparams = {
    "num_epochs": NUM_EPOCHS,
    "total_queries": TOTAL_QUERIES,
    "lr": 1e-2,
}
# posterior_hiding_values = np.linspace(0, 1, HPARAM_GRID_SIZE)  # pH in hiding env
# posterior_default_values = np.linspace(0, 1, HPARAM_GRID_SIZE)  # pD in verbose env
# reward_penalty_values = np.linspace(0, MAX_REWARD_PENALTY, HPARAM_GRID_SIZE)

posterior_default_values = np.linspace(0.3, 0.6, 4)  # pD in verbose env
reward_penalty_values = np.linspace(0, 2, 3)

env_configs = []

# base_env_config_hiding = {
#     "example_name": "hiding_env_example_B1",
#     "env_name": "hiding_env",
#     "hyperparams": {"p": 0.5},
# }
# for pH in posterior_hiding_values:
#     for reward_penalty in reward_penalty_values:
#         env_config = base_env_config_hiding.copy()
#         env_hyperparams = env_config["hyperparams"].copy()
#         env_hyperparams["pH"] = pH
#         env_hyperparams["r"] = reward_penalty
#         env_config["hyperparams"] = env_hyperparams
#         env_configs.append(env_config)

base_env_config_verbose = {
    "example_name": "verbose_env_example_B3",
    "env_name": "verbose_env",
    "hyperparams": {"p": 0.5},
}
for pD in posterior_default_values:
    for reward_penalty in reward_penalty_values:
        env_config = base_env_config_verbose.copy()
        env_hyperparams = env_config["hyperparams"].copy()
        env_hyperparams["pD"] = pD
        env_hyperparams["r"] = reward_penalty
        env_config["hyperparams"] = env_hyperparams
        env_configs.append(env_config)


sweep_config = {
    "method": "grid",
    "parameters": {
        "env": {"values": env_configs},
        "likelihood_model_type": {"value": "naive"},
        "seed": {"values": list(range(NUM_SEEDS))},
        "train_hyperparams": {"value": train_hyperparams},
    },
}


def run_agent(sweep_id):
    wandb.agent(sweep_id, function=experiment.run_condition)


if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)

    num_processes = 4

    processes = []
    for _ in range(num_processes):
        p = Process(target=run_agent, args=(sweep_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
