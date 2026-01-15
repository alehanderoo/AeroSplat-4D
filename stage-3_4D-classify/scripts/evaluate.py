#!/usr/bin/env python3
"""
Evaluation script for 4D Gaussian Classifier.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --data-root data/test
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --sequences seq1 seq2 seq3
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import json
from tqdm import tqdm

from utils.logging import setup_logging, get_logger
from inference.predictor import Predictor
from data.gaussian_dataset import GaussianSequenceDataset, GaussianCollator
from training.metrics import ClassificationMetrics, MetricTracker

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 4D Gaussian Classifier')

    parser.add_argument(
        '--checkpoint', type=Path, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-root', type=Path, default=None,
        help='Root directory containing test sequences'
    )
    parser.add_argument(
        '--sequences', type=str, nargs='+', default=None,
        help='Specific sequence directories to evaluate'
    )
    parser.add_argument(
        '--output', type=Path, default=None,
        help='Output file for results (JSON)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to run evaluation on'
    )
    parser.add_argument(
        '--batch-size', type=int, default=8,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Classification threshold'
    )
    parser.add_argument(
        '--find-threshold', action='store_true',
        help='Find optimal threshold on the data'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print per-sequence results'
    )

    return parser.parse_args()


def evaluate_dataset(
    predictor: Predictor,
    dataset: GaussianSequenceDataset,
    batch_size: int = 8,
    threshold: float = 0.5,
    verbose: bool = False,
):
    """
    Evaluate on a full dataset.

    Returns:
        Dictionary with metrics and per-sequence results
    """
    collator = GaussianCollator()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
    )

    tracker = MetricTracker()
    per_sequence = []

    for batch in tqdm(loader, desc="Evaluating"):
        result = predictor.predict_from_tensors(
            positions=batch['positions'],
            eigenvectors=batch['eigenvectors'],
            scalars=batch['scalars'],
            mask=batch['mask'],
        )

        probs = result['probability']
        if isinstance(probs, float):
            probs = [probs]

        labels = batch['labels'].tolist()
        preds = [(p > threshold) for p in probs]

        tracker.update(preds, labels, probs)

        # Store per-sequence results
        for i, name in enumerate(batch.get('sequence_names', [])):
            per_sequence.append({
                'name': name,
                'label': labels[i],
                'prediction': int(preds[i]),
                'probability': float(probs[i]),
            })

            if verbose:
                status = '✓' if preds[i] == labels[i] else '✗'
                logger.info(f"{status} {name}: pred={preds[i]}, label={labels[i]}, prob={probs[i]:.3f}")

    metrics = tracker.compute()
    return {
        'metrics': metrics,
        'per_sequence': per_sequence,
    }


def main():
    args = parse_args()

    setup_logging(level='INFO')

    # Device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'

    # Load predictor
    logger.info(f"Loading model from {args.checkpoint}")
    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model_info = predictor.get_model_info()
    logger.info(f"Model has {model_info['total_parameters']:,} parameters")

    # Create dataset
    if args.data_root:
        dataset = GaussianSequenceDataset(
            data_root=args.data_root,
            sequence_length=30,  # Will be loaded from checkpoint config
            stride=1,
            max_gaussians=50000,
        )
        logger.info(f"Loaded {len(dataset)} sequences from {args.data_root}")

    elif args.sequences:
        # Evaluate specific sequences
        logger.info(f"Evaluating {len(args.sequences)} sequences")
        results = []

        for seq_path in args.sequences:
            seq_path = Path(seq_path)
            if not seq_path.exists():
                logger.warning(f"Sequence not found: {seq_path}")
                continue

            ply_files = sorted(seq_path.glob("*.ply"))
            if len(ply_files) == 0:
                logger.warning(f"No PLY files in {seq_path}")
                continue

            result = predictor.predict_sequence(ply_files)
            results.append({
                'name': seq_path.name,
                'prediction': result['prediction'],
                'probability': result['probability'],
                'confidence': result['confidence'],
                'latency_ms': result['latency_ms'],
            })

            class_name = 'drone' if result['prediction'] == 1 else 'bird'
            logger.info(
                f"{seq_path.name}: {class_name} "
                f"(prob={result['probability']:.3f}, conf={result['confidence']:.3f}, "
                f"latency={result['latency_ms']:.1f}ms)"
            )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        return

    else:
        logger.error("Either --data-root or --sequences must be specified")
        sys.exit(1)

    # Evaluate dataset
    results = evaluate_dataset(
        predictor=predictor,
        dataset=dataset,
        batch_size=args.batch_size,
        threshold=args.threshold,
        verbose=args.verbose,
    )

    metrics = results['metrics']

    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    if 'auc' in metrics:
        logger.info(f"  AUC-ROC:   {metrics['auc']:.4f}")
    if 'ap' in metrics:
        logger.info(f"  AP:        {metrics['ap']:.4f}")
    logger.info("=" * 50)

    # Find optimal threshold
    if args.find_threshold:
        probs = [r['probability'] for r in results['per_sequence']]
        labels = [r['label'] for r in results['per_sequence']]

        metrics_calc = ClassificationMetrics()
        opt_threshold, opt_f1 = metrics_calc.find_optimal_threshold(probs, labels, 'f1')

        logger.info(f"Optimal threshold: {opt_threshold:.3f} (F1={opt_f1:.4f})")

    # Confusion matrix
    preds = [r['prediction'] for r in results['per_sequence']]
    labels = [r['label'] for r in results['per_sequence']]

    metrics_calc = ClassificationMetrics()
    cm = metrics_calc.compute_confusion_matrix(preds, labels)
    logger.info(f"Confusion Matrix:")
    logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # Save results
    if args.output:
        output_data = {
            'metrics': {k: float(v) for k, v in metrics.items()},
            'confusion_matrix': cm.tolist(),
            'threshold': args.threshold,
            'per_sequence': results['per_sequence'],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
