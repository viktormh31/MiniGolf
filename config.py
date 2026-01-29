
# ----------------
#Environment
#-----------------
TRAIN_ENV_CONFIG= {

    'GUI' : False,
    'reward_type' : "sparse",
}
TEST_ENV_CONFIG = {
     'GUI' : True,
     'reward_type' : "sparse", 
}


# ----------------
#W&B
#-----------------

WANDB_CONFIG= {
    "project": "Xarm-golf",
    "run_id": "test44-optimization",
    "hyperparameters": {
        "lr_actor": 1e-4,
        "lr_critic": 3e-3,
        "batch_size": 2048,
        "nn_dims": 512,
        "temperature": 0.3,
        "episodes": 30000,
        "entropy": -2,
        "init_weights": "xavier_uniform",
        "optimizer": "AdamW",
        "gamma": 0.95,
    }
}


# ----------------
#Agent (SAC)
#-----------------

AGENT_CONFIG = {
     "lr_actor": 1e-4,
    "lr_critic": 3e-3,
    "input_dims": 19,
    "obs_dims": 13,
    "n_actions": 4,
    "max_action": 1,
    "fc1_dim": 512,
    "fc2_dim": 512,
    "batch_size": 2048,
    "gamma": 0.95,
}

# ----------------
#Training
#-----------------

TRAINING_CONFIG = {
    "episode_length": 50,
    "warm_up_phase": 50,
    "num_episodes": 30000,
    "phase_1_threshold": 3000,
    "phase_2_threshold": 10000,
}

PHASE_SWITCH ={
    3000: 2,
    10000: 3,
}


