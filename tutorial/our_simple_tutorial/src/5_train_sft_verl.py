import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf

# Add the current working directory to sys.path to ensure local modules can be imported correctly
sys.path.append(os.getcwd())

# Import the standard SFT trainer main function from the verl library
from verl.trainer.fsdp_sft_trainer import main as run_sft_trainer

@hydra.main(config_path="../config", config_name="sft", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for SFT (Supervised Fine-Tuning) training using VERL.
    
    Args:
        cfg (DictConfig): The configuration object loaded by Hydra from config/sft.yaml.
    """
    # Print the configuration for verification on the main process (rank 0)
    if cfg.trainer.get("local_rank", 0) == 0:
        print("Loading SFT Configuration:\n", OmegaConf.to_yaml(cfg))
    
    try:
        # Execute the standard SFT training pipeline provided by verl
        run_sft_trainer(cfg)
    except Exception as e:
        print(f"[Error] SFT Training failed: {e}")
        raise e

if __name__ == "__main__":
    main()
