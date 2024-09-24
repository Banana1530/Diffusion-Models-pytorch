import torch
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.run_name = "DDPM_conditional"
torch.save(120, os.path.join("models", "DDPM_conditional", f"epoch_ckpt.pt"))

torch.load(os.path.join("models", "DDPM_conditional", f"epoch_ckpt.pt"))