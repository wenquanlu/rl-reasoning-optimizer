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
