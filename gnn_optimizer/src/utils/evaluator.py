import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Evaluator:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        
    def log_epoch(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def calculate_metrics(self, predictions, targets):
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    def plot_learning_curves(self, save_path="experiments/plots/loss_curve.png"):

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', color='blue', marker='o')
        plt.plot(self.val_losses, label='Validation Loss', color='orange', marker='o')
        plt.title('Self-Supervised Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f" Saved Learning Curve to {save_path}")

    def plot_predictions_vs_truth(self, predictions, targets, feature_name="Queue Length", save_path="experiments/plots/scatter.png"):

        if isinstance(predictions, torch.Tensor): predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor): targets = targets.detach().cpu().numpy()

        # Flatten if multidimensional
        preds = predictions.flatten()
        actuals = targets.flatten()
        
        # Randomly sample points if too many (to avoid messy plot)
        if len(preds) > 2000:
            indices = np.random.choice(len(preds), 2000, replace=False)
            preds = preds[indices]
            actuals = actuals[indices]

        plt.figure(figsize=(8, 8))
        plt.scatter(actuals, preds, alpha=0.5, s=10, color='purple')
        
        # Draw perfect prediction line (y=x)
        min_val = min(actuals.min(), preds.min())
        max_val = max(actuals.max(), preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.title(f'Actual vs Predicted: {feature_name}')
        plt.xlabel(f'Actual {feature_name}')
        plt.ylabel(f'Predicted {feature_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f" Saved Prediction Scatter Plot to {save_path}")

    def plot_time_series_sample(self, predictions, targets, sample_length=100, save_path="experiments/plots/timeseries.png"):
        if isinstance(predictions, torch.Tensor): predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor): targets = targets.detach().cpu().numpy()

        # Simple flatten for visualization:
        subset_preds = predictions.flatten()[:sample_length]
        subset_actuals = targets.flatten()[:sample_length]

        plt.figure(figsize=(12, 5))
        plt.plot(subset_actuals, label='Actual Reality', color='black', linestyle='-', alpha=0.7)
        plt.plot(subset_preds, label='GNN Prediction', color='green', linestyle='--')
        
        plt.title(f'Traffic Flow Prediction Sample (First {sample_length} data points)')
        plt.xlabel('Time Step (Seconds)')
        plt.ylabel('Traffic Value (Normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()
        print(f" Saved Time-Series Plot to {save_path}")

    def plot_error_distribution(self, predictions, targets, save_path="experiments/plots/error_hist.png"):
        if isinstance(predictions, torch.Tensor): predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor): targets = targets.detach().cpu().numpy()

        errors = predictions.flatten() - targets.flatten()
        
        plt.figure(figsize=(8, 6))
        plt.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
        plt.title('Error Distribution (Residuals)')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.savefig(save_path)
        plt.close()
        print(f" Saved Error Histogram to {save_path}")

    def plot_marl_performance(self, rewards, queues, losses, save_dir="experiments/plots"):
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        episodes = range(1, len(rewards) + 1)
        
        # 1. Reward Curve
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards, marker='o', color='green', label='Total Reward')
        plt.title('MARL Learning Curve: Cumulative Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.savefig(f"{save_dir}/marl_reward_curve.png")
        plt.close()
        
        # 2. Queue Length Curve
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, queues, marker='s', color='red', label='Avg Queue Length')
        plt.title('Traffic Congestion: Average Queue Length')
        plt.xlabel('Episode')
        plt.ylabel('Vehicles per Lane')
        plt.grid(True)
        plt.savefig(f"{save_dir}/marl_queue_curve.png")
        plt.close()

        # 3. Loss Curve (Optional but good for debugging)
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, losses, marker='x', color='blue', label='Actor Loss')
        plt.title('Training Stability: Actor Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{save_dir}/marl_loss_curve.png")
        plt.close()
        
        print(f" Saved MARL performance plots to {save_dir}")