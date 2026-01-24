import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf

# Add the current working directory to sys.path to allow importing local modules
sys.path.append(os.getcwd())

# --- CRITICAL IMPORT SECTION ---
# We must import the custom reward definition module BEFORE the trainer starts.
# This import triggers the @registry.register decorators in 'src.reward_utils',
# allowing the verl config system to locate your custom reward function (e.g., 'medical_compliance').
try:
    import src.reward_utils 
except ImportError:
    print("[Warning] Could not import 'src.reward_utils'. Ensure the file exists and sys.path is correct.")
# -------------------------------

# Import the main PPO/GRPO training entry point from verl
# Note: verl uses the same 'main_ppo' entry point for both PPO and GRPO algorithms.
# The specific algorithm (GRPO) is determined by the config file (algorithm.adv_estimator: grpo).
from verl.trainer.main_ppo import main as run_ppo_trainer

@hydra.main(config_path="../config", config_name="grpo", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for GRPO training using VERL.
    
    This wrapper ensures that custom reward functions are registered 
    before the reinforcement learning pipeline is initialized.
    """
    # Print configuration info on the main process
    if cfg.trainer.get("local_rank", 0) == 0:
        print("Loading GRPO Configuration:\n", OmegaConf.to_yaml(cfg))
        print(f"Active Reward Function: {cfg.data.reward_metric}")

    try:
        # Execute the Reinforcement Learning pipeline (supports Multi-GPU/FSDP/Megatron)
        run_ppo_trainer(cfg)
    except Exception as e:
        print(f"[Error] GRPO Training failed: {e}")
        raise e

if __name__ == "__main__":
    main()
