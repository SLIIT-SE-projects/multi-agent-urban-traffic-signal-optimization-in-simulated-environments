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