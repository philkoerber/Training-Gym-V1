"""Lightning App wrapper for cloud training."""

import lightning as L
from lightning_app import LightningApp, LightningWork, CloudCompute
from alpaca.data.timeframe import TimeFrame

from train import train


class TrainingWork(LightningWork):
    """Lightning Work component that runs training on cloud GPU."""
    
    def __init__(
        self,
        days: int = 1460,
        seq_len: int = 60,
        pred_len: int = 1,
        model_type: str = "lstm",
        hidden_dim: int = 64,
        num_layers: int = 2,
        batch_size: int = 32,
        max_epochs: int = 50,
        lr: float = 1e-3,
        timeframe: str = "minute",
        num_workers: int = 7,
        **kwargs
    ):
        # Configure cloud compute - use GPU
        cloud_compute = CloudCompute("gpu", name="gpu-fast")
        super().__init__(cloud_compute=cloud_compute, **kwargs)
        
        # Store training parameters
        self.days = days
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.timeframe_str = timeframe
        self.num_workers = num_workers
    
    def run(self):
        """Execute training with cloud GPU."""
        # Convert timeframe string to enum
        tf = TimeFrame.Minute if self.timeframe_str == "minute" else TimeFrame.Hour
        
        print(f"\n{'='*50}")
        print("Running training on Lightning AI Cloud GPU")
        print(f"{'='*50}\n")
        
        # Call the training function with use_cloud=True
        train(
            days=self.days,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            model_type=self.model_type,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            lr=self.lr,
            timeframe=tf,
            use_cloud=True,  # Always use cloud when running as Lightning App
            num_workers=self.num_workers,
        )


class TrainingApp(LightningApp):
    """Main Lightning App for cloud training."""
    
    def __init__(
        self,
        days: int = 1460,
        seq_len: int = 60,
        pred_len: int = 1,
        model_type: str = "lstm",
        hidden_dim: int = 64,
        num_layers: int = 2,
        batch_size: int = 32,
        max_epochs: int = 50,
        lr: float = 1e-3,
        timeframe: str = "minute",
        num_workers: int = 7,
    ):
        work = TrainingWork(
            days=days,
            seq_len=seq_len,
            pred_len=pred_len,
            model_type=model_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_size=batch_size,
            max_epochs=max_epochs,
            lr=lr,
            timeframe=timeframe,
            num_workers=num_workers,
        )
        super().__init__(work)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train on Lightning AI Cloud")
    parser.add_argument("--days", type=int, default=1460)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timeframe", type=str, default="minute", choices=["minute", "hour"])
    parser.add_argument("--num_workers", type=int, default=7)
    
    args = parser.parse_args()
    
    app = TrainingApp(
        days=args.days,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        timeframe=args.timeframe,
        num_workers=args.num_workers,
    )
    
    app.run()

