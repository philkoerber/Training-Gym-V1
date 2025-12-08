"""Lightning App wrapper for cloud training.

This app reads training configuration from train.py and runs training on Lightning AI cloud GPUs.
All training parameters are configured in train.py at the top of the file.
"""

from lightning_app import LightningApp, LightningWork, CloudCompute

from .train import train


class TrainingWork(LightningWork):
    """Lightning Work component that runs training on cloud GPU."""
    
    def __init__(self, **kwargs):
        # Configure cloud compute - use GPU
        # Valid names: "default", "gpu", "gpu-fast", "cpu", etc.
        cloud_compute = CloudCompute("gpu-fast")
        super().__init__(cloud_compute=cloud_compute, **kwargs)
    
    def run(self):
        """Execute training with cloud GPU."""
        print(f"\n{'='*50}")
        print("Running training on Lightning AI Cloud GPU")
        print("Training parameters are read from train.py configuration")
        print(f"{'='*50}\n")
        
        # Call the training function - all parameters come from train.py config
        train(use_cloud=True)


class TrainingApp(LightningApp):
    """Main Lightning App for cloud training."""
    
    def __init__(self):
        work = TrainingWork()
        super().__init__(work)


# Export the app for Lightning CLI
app = TrainingApp()

