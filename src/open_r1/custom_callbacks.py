import os
import glob
from transformers import TrainerCallback


class OptimStateCleanupCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        """
        Deletes optimizer and model state files from:
          - All `checkpoint-*` folders
          - The top-level `output_dir`
        Patterns deleted:
          - *_optim_states.pt (DeepSpeed)
          - *_model_states.pt (DeepSpeed)
          - optimizer.pt      (DDP)
          - scheduler.pt      (optional, training-only)
        """
        deleted_count = 0

        # Patterns to match
        patterns = [
            "*_optim_states.pt",
            "*_model_states.pt",
            "optimizer.pt"
        ]

        # === 1. Delete from all checkpoint-* folders ===
        checkpoint_dirs = sorted(
            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
            key=os.path.getctime,
            reverse=True,
        )

        for ckpt in checkpoint_dirs:
            for pattern in patterns:
                for f in glob.glob(os.path.join(ckpt, pattern)):
                    try:
                        os.remove(f)
                        print(f"[Cleanup] Deleted from {ckpt}: {f}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"[Cleanup] Error deleting {f}: {e}")

        # === 2. Delete from top-level output_dir ===
        for pattern in patterns:
            for f in glob.glob(os.path.join(args.output_dir, pattern)):
                try:
                    os.remove(f)
                    print(f"[Cleanup] Deleted from output_dir: {f}")
                    deleted_count += 1
                except Exception as e:
                    print(f"[Cleanup] Error deleting {f}: {e}")

        if deleted_count > 0:
            print(f"[Cleanup] Deleted {deleted_count} optimizer/model state files.")
        else:
            print(f"[Cleanup] No optimizer/model state files found.")

        return control

class ForceEvalCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_evaluate = True
        return control


class GradientMonitorCallback(TrainerCallback):
    def __init__(self):
        self.grad_running_mean = None
        self.grad_running_mean_squared = None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        return
        model = kwargs["model"]
        accelerator = kwargs["accelerator"]

        # this is step before current step, because step increments after this callback in trainer
        step = state.global_step

        # Collect all gradients in a single flattened vector
        grads = [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return  # Skip if no grads this step

        flat_grad = torch.cat(grads)
        grad_norm = torch.norm(flat_grad, p=2).item()
        grad_var = torch.var(flat_grad).item()

        # Initialize or update running stats
        if self.grad_running_mean is None:
            self.grad_running_mean = flat_grad.clone()
            self.grad_running_mean_squared = flat_grad.clone() ** 2
        else:
            self.grad_running_mean = (self.grad_running_mean * step + flat_grad) / (step + 1)
            self.grad_running_mean_squared = (self.grad_running_mean_squared * step + flat_grad ** 2) / (step + 1)

        # Reduce stats across processes (mean reduction)
        grad_norm_tensor = torch.tensor(grad_norm, device=accelerator.device)
        grad_var_tensor = torch.tensor(grad_var, device=accelerator.device)

        grad_norm_tensor = accelerator.reduce(grad_norm_tensor, reduction="mean")
        grad_var_tensor = accelerator.reduce(grad_var_tensor, reduction="mean")
        flat_grad = accelerator.reduce(flat_grad, reduction="mean")
        self.grad_running_mean = accelerator.reduce(self.grad_running_mean, reduction="mean")
        self.grad_running_mean_squared = accelerator.reduce(self.grad_running_mean_squared, reduction="mean")
        if step + 1 >= 10:
            grad_std = (self.grad_running_mean_squared - self.grad_running_mean ** 2).sqrt()
            lambda_sigma = 3.0
            deviation = (flat_grad - self.grad_running_mean).abs()
            outliers = (deviation > lambda_sigma * grad_std)
            proportion_outliers = outliers.float().mean().item()


        if accelerator.is_main_process:
            info = {
                "grad/post_clip_norm": grad_norm_tensor.item(),
                "grad/variance": grad_var_tensor.item(),
            }
            if step + 1 >= 10:
                info["grad/proportion_spike"] = proportion_outliers
            #print(f"[Step Debug] HF global_step: {state.global_step}, wandb.run.step: {wandb.run.step}")
            wandb.log(info, step=wandb.run.step + 1)
            #print(f"[Step {step + 1}] Pre-clip grad norm: {grad_norm_tensor.item():.4f} | Var: {grad_var_tensor.item():.4f}")