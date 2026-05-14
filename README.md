# LiDARS

Structured change prediction for longitudinal chest X-ray interpretation.
Given patient studies, the model outputs a delta report: NEW, RESOLVED, and STABLE_PRESENT findings.

Built on PadChest. Uses ResNet-18 + Liquid Neural Network conditioned on inter-study time gap.

## Setup
pip install torch torchvision pandas scikit-learn tqdm pillow matplotlib

## Data
Request PadChest access: https://bimcv.cipf.es/bimcv-projects/padchest/

## Checkpoint
Download from Releases and place at artifacts/delta_runs/best.pt

## License
MIT
