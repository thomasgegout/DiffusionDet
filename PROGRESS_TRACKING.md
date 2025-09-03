# Training Progress Enhancement with tqdm

The `train_accelerate.py` script has been enhanced with comprehensive progress tracking using tqdm progress bars.

## New Features

### 1. Main Training Progress Bar
- Shows overall training progress with percentage completion
- Displays current step, total steps, and percentage complete
- Real-time metrics: loss, learning rate, training speed (steps/sec), current epoch
- Estimated time remaining (ETA) calculation
- Dynamic updates with colorful progress visualization

### 2. Enhanced Evaluation Progress Bar
- Separate progress bar for validation/evaluation phases
- Shows validation loss and running average
- Automatically hides after completion (leave=False)
- Clear indication when switching between training and evaluation

### 3. Comprehensive Timing Metrics
- **Step timing**: Individual step execution time tracking
- **Speed calculation**: Steps per second with rolling average
- **ETA estimation**: Estimated time to completion based on recent performance
- **Total elapsed time**: Running total of training time
- **Training summary**: Final statistics at training completion

### 4. Enhanced Logging Output

#### Console Output Format:
```
Step 1000/5000 (20.0%) - Time: 0h 15m 30s - Speed: 2.35 steps/s - ETA: 1h 2m 15s - train_loss=0.234567 learning_rate=1.50e-04 epoch=2
```

#### Progress Bar Display:
```
Training: 20%|██▎        | 1000/5000 [15:30<62:15, 2.35step/s, loss=0.2346, lr=1.50e-04, step/s=2.35, epoch=2]
```

#### Evaluation Display:
```
Evaluating: 100%|████████████| 50/50 [00:45<00:00, 1.11batch/s, val_loss=0.1987, avg_loss=0.2012]
```

### 5. Training Summary
At the end of training, a comprehensive summary is displayed:
```
Training Summary:
  Total steps: 5000
  Total time: 2h 15m 45s
  Average step time: 1.623s
  Average steps per second: 0.62
  Epochs completed: 8
```

## Usage

The enhanced progress tracking is automatically enabled when running the training script:

```bash
python train_accelerate.py \
    --config-file configs/diffdet.tables.res50.cpu.yaml \
    --output-dir ./output_accelerate \
    --device cpu \
    --gradient-accumulation-steps 2
```

## Visual Features

### Progress Bar Elements:
- **Progress bar**: Visual representation of completion percentage
- **Percentage**: Exact completion percentage (e.g., 34.2%)
- **Step counter**: Current step / total steps (e.g., 1710/5000)
- **Elapsed time**: Time since training started (e.g., [25:30<49:15])
- **ETA**: Estimated time remaining
- **Speed**: Current processing speed (e.g., 1.23step/s)
- **Live metrics**: Current loss, learning rate, epoch number

### Status Indicators:
- **Training**: Green progress bar during normal training
- **Evaluating**: Blue progress bar during validation
- **Saving checkpoint**: Yellow indicator during checkpoint saves

## Performance Monitoring

The enhanced logging provides detailed performance insights:

1. **Real-time speed tracking**: Monitor training throughput
2. **Performance trends**: Track if training is slowing down over time
3. **Memory usage patterns**: Observe correlation between progress and system performance
4. **Training efficiency**: Identify bottlenecks in the training pipeline

## MLflow Integration

All timing metrics are also logged to MLflow (when enabled):
- `steps_per_second`: Training throughput
- `elapsed_time_hours`: Total training time
- `step_time`: Individual step execution time
- Standard training metrics (loss, learning rate, etc.)

## Requirements

The enhanced progress tracking requires:
```
tqdm>=4.60.0
```

This dependency is included in `requirements_accelerate.txt`.

## Benefits

1. **Better user experience**: Clear visual feedback on training progress
2. **Performance monitoring**: Real-time speed and efficiency metrics
3. **Time management**: Accurate ETA calculations for planning
4. **Debugging assistance**: Identify slow steps or performance issues
5. **Professional appearance**: Clean, colorful progress visualization
6. **Interrupt handling**: Graceful cleanup of progress bars on interruption

## Technical Implementation

- **Thread-safe**: Works correctly with multi-process training
- **Main process only**: Progress bars only shown on the main process (rank 0)
- **Memory efficient**: Rolling window for step timing (keeps last 100 steps)
- **Graceful cleanup**: Proper progress bar cleanup on interruption or completion
- **Dynamic updates**: Real-time metric updates without performance impact
