"""
Training hooks for logging and callbacks
"""
import time
import os
from datetime import datetime
from typing import Dict, Optional
from beautifultable import BeautifulTable


class TextLoggerHook:
    """
    Hook for logging training information at specified intervals
    """
    
    def __init__(
        self,
        log_interval: int = 10,
        print_lr: bool = True,
        print_time: bool = True,
        log_file: Optional[str] = None
    ):
        """
        Args:
            log_interval: Print logs every N iterations
            print_lr: Whether to print learning rate
            print_time: Whether to print elapsed time
            log_file: Path to log file (if None, no file logging)
        """
        self.log_interval = log_interval
        self.print_lr = print_lr
        self.print_time = print_time
        self.log_file = log_file
        self.start_time = None
        self.last_log_time = None
        self.iter_count = 0
        self.epoch_count = 0
        
        # Open log file if specified
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            self.log_fp = open(self.log_file, 'w')
            self.log_fp.write(f"Training Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_fp.write("="*80 + "\n\n")
        else:
            self.log_fp = None
    
    def _log(self, message: str, print_to_console: bool = True):
        """Log message to both console and file"""
        if print_to_console:
            print(message)
        if self.log_fp:
            self.log_fp.write(message + "\n")
            self.log_fp.flush()
    
    def before_train(self):
        """Called before training starts"""
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.iter_count = 0
        self._log("\n" + "="*60)
        self._log("Starting Training")
        self._log("="*60)
    
    def before_epoch(self, epoch: int, total_epochs: int):
        """Called before each epoch"""
        self.epoch_count = epoch
        self.iter_count = 0
        self.last_log_time = time.time()
        self._log(f"\nEpoch {epoch}/{total_epochs}")
        self._log("-" * 60)
    
    def after_iter(
        self,
        iter_idx: int,
        total_iters: int,
        loss: float,
        lr: Optional[float] = None,
        metrics: Optional[Dict] = None,
        loss_components: Optional[Dict] = None
    ):
        """
        Called after each iteration
        
        Args:
            iter_idx: Current iteration index (0-based)
            total_iters: Total iterations in current epoch
            loss: Current loss value
            lr: Current learning rate
            metrics: Optional dictionary of additional metrics
        """
        self.iter_count = iter_idx + 1
        
        # Print at specified intervals or at the end of epoch
        if (self.iter_count % self.log_interval == 0) or (self.iter_count == total_iters):
            current_time = time.time()
            elapsed_time = current_time - self.last_log_time
            total_elapsed = current_time - self.start_time if self.start_time else 0
            
            # Build log message with loss components if available
            if loss_components is not None and len(loss_components) > 0:
                # Show individual loss components
                loss_parts = []
                for key, value in loss_components.items():
                    if not key.endswith('_weighted'):  # Only show unweighted components
                        loss_parts.append(f"{key.upper()}: {value:.4f}")
                loss_str = ", ".join(loss_parts) + f" | Loss: {loss:.4f}"
            else:
                loss_str = f"Loss: {loss:.4f}"
            
            log_parts = [
                f"[Epoch {self.epoch_count}]",
                f"Iter [{self.iter_count}/{total_iters}]",
                loss_str
            ]
            
            # Add learning rate
            if self.print_lr and lr is not None:
                log_parts.append(f"LR: {lr:.2e}")
            
            # Add time information
            if self.print_time:
                log_parts.append(f"Time: {elapsed_time:.2f}s")
                if total_elapsed > 0:
                    eta = (total_elapsed / self.iter_count) * (total_iters - self.iter_count)
                    log_parts.append(f"ETA: {eta:.0f}s")
            
            # Add additional metrics
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, float):
                        log_parts.append(f"{key}: {value:.4f}")
                    else:
                        log_parts.append(f"{key}: {value}")
            
            self._log(" | ".join(log_parts))
            self.last_log_time = current_time
    
    def after_epoch(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Optional[Dict] = None,
        lr: Optional[float] = None
    ):
        """
        Called after each epoch
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary (optional)
            lr: Current learning rate
        """
        self._log("\n" + "="*60)
        self._log(f"Epoch {epoch} Summary")
        self._log("="*60)
        
        # Print training metrics
        self._log("\nTRAINING METRICS")
        self._log("-" * 60)
        train_table = BeautifulTable()
        
        # Build header and row data
        header = ["Loss", "RMSLE (Primary)", "RMSE", "MAE", "R²"]
        row_data = [
            f"{train_metrics['loss']:.4f}",
            f"{train_metrics['rmsle']:.4f}",
            f"{train_metrics['rmse']:.4f}",
            f"{train_metrics['mae']:.4f}",
            f"{train_metrics.get('r2', 0):.4f}"
        ]
        
        # Add MAPE if available
        if 'mape' in train_metrics:
            header.append("MAPE (%)")
            row_data.append(f"{train_metrics['mape']:.2f}")
        
        train_table.columns.header = header
        train_table.rows.append(row_data)
        self._log(str(train_table))
        
        # Print validation metrics if available
        if val_metrics:
            self._log("\nVALIDATION METRICS")
            self._log("-" * 60)
            val_table = BeautifulTable()
            
            # Build header and row data
            header = ["Loss", "RMSLE (Primary)", "RMSE", "MAE", "R²"]
            row_data = [
                f"{val_metrics['loss']:.4f}",
                f"{val_metrics['rmsle']:.4f}",
                f"{val_metrics['rmse']:.4f}",
                f"{val_metrics['mae']:.4f}",
                f"{val_metrics.get('r2', 0):.4f}"
            ]
            
            # Add MAPE if available
            if 'mape' in val_metrics:
                header.append("MAPE (%)")
                row_data.append(f"{val_metrics['mape']:.2f}")
            
            val_table.columns.header = header
            val_table.rows.append(row_data)
            self._log(str(val_table))
            
            # Print sample predictions if available
            if 'sample_predictions' in val_metrics and len(val_metrics['sample_predictions']) > 0:
                self._log("\nSAMPLE PREDICTIONS (First 5 Validation Samples)")
                self._log("-" * 60)
                sample_table = BeautifulTable()
                sample_table.columns.header = ["Sample ID", "Ground Truth", "Prediction", "Error", "Error %"]
                
                for i in range(min(5, len(val_metrics['sample_predictions']))):
                    sample_id = val_metrics['sample_ids'][i] if i < len(val_metrics['sample_ids']) else f"Sample_{i+1}"
                    gt = val_metrics['sample_targets'][i]
                    pred = val_metrics['sample_predictions'][i]
                    error = abs(pred - gt)
                    error_pct = (error / (gt + 1e-8)) * 100
                    
                    sample_table.rows.append([
                        str(sample_id),
                        f"{gt:.2f}",
                        f"{pred:.2f}",
                        f"{error:.2f}",
                        f"{error_pct:.2f}%"
                    ])
                
                self._log(str(sample_table))
        
        # Print learning rate if enabled
        if self.print_lr and lr is not None:
            self._log(f"\nLearning Rate: {lr:.2e}")
        
        self._log("="*60 + "\n")
    
    def after_train(self):
        """Called after training completes"""
        if self.start_time:
            total_time = time.time() - self.start_time
            self._log("="*60)
            self._log(f"Training completed in {total_time/60:.2f} minutes")
            self._log("="*60)
        
        # Close log file
        if self.log_fp:
            self.log_fp.write(f"\nTraining Log - Ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_fp.close()

