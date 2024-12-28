from multiprocessing import Process

# import porlhf_paper_experiment as experiment
import porlhf_paper_experiment_logging_per_epoch as experiment
import wandb

# Configure experiment

PROJECT_NAME = "porlhf_paper_experiment_debug"

TOTAL_QUERIES = 378
NUM_EPOCHS = 300
NUM_SEEDS = 10

train_hyperparams = {
    "num_epochs": NUM_EPOCHS,
    "total_queries": TOTAL_QUERIES,
    "lr": 5e-3,
}
likelihood_model_types = [
    "naive",
    "po-aware",
]
envs = [
    {
        "example_name": "hiding_env_example_B1",
        "env_name": "hiding_env",
        "hyperparams": {"p": 0.5, "pH": 0.5},
        "description": "In Environment A, a deceptive inflation example (Example B.1)."
        "Optimal action is aC."
        "Theory predicts naive agent will take aH; PO-aware agent will have ambiguity.",
    },
    {
        "example_name": "hiding_env_example_B2",
        "env_name": "hiding_env",
        "hyperparams": {"p": 0.1, "pH": 0.9},
        "description": "In Environment A, an overjustification example (Example B.2)."
        "Optimal action is aT."
        "Theory predicts naive agent will take aC; PO-aware agent will have ambiguity.",
    },
    {
        "example_name": "verbose_env_example_B3_1",
        "env_name": "verbose_env",
        "hyperparams": {"p": 0.5, "pD": 0.9},
        "description": "In Environment B, a deceptive inflation example (Example B.3, first paragraph)."
        "Optimal action is aD."
        "Theory predicts naive agent will take aT; PO-aware agent will take aD.",
    },
    {
        "example_name": "verbose_env_example_B3_2",
        "env_name": "verbose_env",
        "hyperparams": {"p": 0.5, "pD": 0.1},
        "description": "In Environment B, an overjustification example (Example B.3, second paragraph)."
        "Optimal action is aD."
        "Theory predicts naive agent will take aV; PO-aware agent will take aD.",
    },
    # {
    #     "example_name": "verbose_env_example_B3_2_large_p",
    #     "env_name": "verbose_env",
    #     "hyperparams": {"p": 0.9, "pD": 0.1},
    #     "description": "In Environment B, an overjustification example (Example B.3, second paragraph)."
    #                    "Optimal action is aD."
    #                    "Theory predicts naive agent will take aV; PO-aware agent will take aD.",
    # },
    # {
    #     "example_name": "verbose_env_example_B3_2_small_p",
    #     "env_name": "verbose_env",
    #     "hyperparams": {"p": 0.35, "pD": 0.1},
    #     "description": "In Environment B, an overjustification example (Example B.3, second paragraph)."
    #                    "Optimal action is aD."
    #                    "Theory predicts naive agent will take aV; PO-aware agent will take aD.",
    # },
]

sweep_config = {
    "method": "grid",
    "parameters": {
        "env": {"values": envs},
        "likelihood_model_type": {"values": likelihood_model_types},
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
