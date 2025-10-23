"""
Train UMAP model on base embeddings to create the stable 2D music universe map.

This creates the fixed coordinate system that all users' listening histories
will be projected onto.
"""

import logging
from pathlib import Path
import argparse
import numpy as np
import umap
import joblib
import matplotlib.pyplot as plt
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from setup.database import EmbeddingDatabase

# Set up logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UMAPTrainer:
    """Train and save UMAP model for music embedding projection."""

    def __init__(self):
        """Initialize the trainer."""
        self.db = EmbeddingDatabase()
        self.umap_model = None
        self.track_ids = None
        self.embeddings = None
        self.projected_coords = None

    def load_embeddings(self):
        """Load all embeddings from database."""
        logger.info("Loading embeddings from database...")
        self.track_ids, self.embeddings = self.db.get_all_embeddings()

        logger.info(f"Loaded {len(self.track_ids)} embeddings")
        logger.info(f"Embedding matrix shape: {self.embeddings.shape}")

        if len(self.track_ids) == 0:
            raise ValueError("No embeddings found in database. Run generate_embeddings.py first.")

        return self.embeddings

    def train_umap(self,
                   n_neighbors: int = config.UMAP_N_NEIGHBORS,
                   min_dist: float = config.UMAP_MIN_DIST,
                   n_components: int = config.UMAP_N_COMPONENTS,
                   metric: str = config.UMAP_METRIC,
                   random_state: int = config.UMAP_RANDOM_STATE):
        """
        Train UMAP model on embeddings.

        Args:
            n_neighbors: Number of neighbors to consider for manifold approximation
            min_dist: Minimum distance between points in low-dimensional space
            n_components: Number of dimensions to reduce to (2 for visualization)
            metric: Distance metric to use
            random_state: Random seed for reproducibility
        """
        logger.info("Training UMAP model...")
        logger.info(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, "
                   f"n_components={n_components}, metric={metric}")

        # Initialize UMAP
        self.umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state,
            verbose=True
        )

        # Fit and transform
        logger.info("Fitting UMAP (this may take several minutes for large datasets)...")
        self.projected_coords = self.umap_model.fit_transform(self.embeddings)

        logger.info(f"UMAP training complete. Projected shape: {self.projected_coords.shape}")

        return self.projected_coords

    def save_model(self, model_path: Path = config.UMAP_MODEL_PATH):
        """
        Save the fitted UMAP model.

        CRITICAL: We save the fitted model so we can use transform() later
        for new songs, ensuring they map to the same coordinate system.
        """
        if self.umap_model is None:
            raise ValueError("No UMAP model to save. Train first.")

        model_path.parent.mkdir(exist_ok=True, parents=True)

        logger.info(f"Saving UMAP model to {model_path}...")
        joblib.dump(self.umap_model, model_path)
        logger.info("UMAP model saved successfully")

    def save_coordinates_to_db(self):
        """Save projected 2D coordinates back to database."""
        logger.info("Saving UMAP coordinates to database...")

        coords_dict = {}
        for track_id, coords in zip(self.track_ids, self.projected_coords):
            coords_dict[track_id] = (float(coords[0]), float(coords[1]))

        self.db.update_umap_coords(coords_dict)
        logger.info(f"Saved coordinates for {len(coords_dict)} tracks")

    def visualize_projection(self, output_path: Optional[Path] = None, sample_size: int = None):
        """
        Create a visualization of the 2D projection.

        Args:
            output_path: Where to save the plot (if None, displays interactively)
            sample_size: If provided, plot only a random sample of points
        """
        if self.projected_coords is None:
            raise ValueError("No projection to visualize. Train UMAP first.")

        logger.info("Creating visualization...")

        # Sample if requested (for large datasets)
        if sample_size and len(self.projected_coords) > sample_size:
            indices = np.random.choice(len(self.projected_coords), sample_size, replace=False)
            coords_to_plot = self.projected_coords[indices]
            logger.info(f"Sampling {sample_size} points for visualization")
        else:
            coords_to_plot = self.projected_coords

        # Create plot
        plt.figure(figsize=config.VIZ_FIGSIZE, dpi=config.VIZ_DPI)
        plt.scatter(
            coords_to_plot[:, 0],
            coords_to_plot[:, 1],
            s=1,
            alpha=0.5,
            c=range(len(coords_to_plot)),
            cmap=config.VIZ_CMAP
        )
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title(f'Music Universe Map ({len(self.track_ids)} songs)')
        plt.colorbar(label='Song Index')
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path)
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def analyze_projection(self):
        """Analyze the quality of the projection."""
        if self.projected_coords is None:
            raise ValueError("No projection to analyze. Train UMAP first.")

        logger.info("=" * 60)
        logger.info("Projection Analysis")
        logger.info("=" * 60)

        # Basic statistics
        x_coords = self.projected_coords[:, 0]
        y_coords = self.projected_coords[:, 1]

        logger.info(f"X range: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
        logger.info(f"Y range: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
        logger.info(f"X mean/std: {x_coords.mean():.2f} / {x_coords.std():.2f}")
        logger.info(f"Y mean/std: {y_coords.mean():.2f} / {y_coords.std():.2f}")

        # Check for clustering (using simple density measure)
        # Points should be relatively well-distributed
        distances = np.linalg.norm(self.projected_coords - self.projected_coords.mean(axis=0), axis=1)
        logger.info(f"Distance from center - mean: {distances.mean():.2f}, std: {distances.std():.2f}")

        logger.info("=" * 60)

    def run(self, visualize: bool = True, save: bool = True):
        """
        Run the complete UMAP training pipeline.

        Args:
            visualize: Whether to create visualization
            save: Whether to save model and coordinates
        """
        logger.info("=" * 60)
        logger.info("Starting UMAP training pipeline")
        logger.info("=" * 60)

        # Load embeddings
        self.load_embeddings()

        # Train UMAP
        self.train_umap()

        # Analyze projection
        self.analyze_projection()

        # Save model
        if save:
            self.save_model()
            self.save_coordinates_to_db()

        # Create visualization
        if visualize:
            viz_path = config.VIZ_OUTPUT_DIR / "music_universe_base.png"
            self.visualize_projection(output_path=viz_path, sample_size=10000)

        logger.info("=" * 60)
        logger.info("UMAP training complete!")
        logger.info(f"Model saved to: {config.UMAP_MODEL_PATH}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train UMAP model on music embeddings")
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save model and coordinates (testing only)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Do not create visualization'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=config.UMAP_N_NEIGHBORS,
        help=f'UMAP n_neighbors parameter (default: {config.UMAP_N_NEIGHBORS})'
    )
    parser.add_argument(
        '--min-dist',
        type=float,
        default=config.UMAP_MIN_DIST,
        help=f'UMAP min_dist parameter (default: {config.UMAP_MIN_DIST})'
    )

    args = parser.parse_args()

    # Create trainer
    trainer = UMAPTrainer()

    # Override UMAP parameters if provided
    if args.n_neighbors != config.UMAP_N_NEIGHBORS or args.min_dist != config.UMAP_MIN_DIST:
        logger.info(f"Using custom UMAP parameters: n_neighbors={args.n_neighbors}, min_dist={args.min_dist}")
        config.UMAP_N_NEIGHBORS = args.n_neighbors
        config.UMAP_MIN_DIST = args.min_dist

    # Run pipeline
    trainer.run(
        visualize=not args.no_viz,
        save=not args.no_save
    )


if __name__ == "__main__":
    main()
