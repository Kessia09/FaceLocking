# Face Locking Project

This project extends a face recognition system by adding a Face Locking feature.

## How Face Locking Works
- A selected enrolled identity is manually chosen
- Once confidently recognized, the system locks onto that face
- The locked face is tracked across frames
- Brief recognition failures are tolerated
- Lock is released only after sustained disappearance

## Actions Detected
- Face movement left
- Face movement right
- Eye blink (simple logic)
- Smile / laugh (simple ratio-based logic)

## Action History
Actions are recorded in files named:

<face>_history_<timestamp>.txt

Each entry includes:
- Timestamp
- Action type
- Description

## Constraints
- CPU-only execution
- No model retraining
- Existing recognition pipeline preserved
