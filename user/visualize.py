"""
Create visualizations of user's listening journey on the music map.

Generates 2D scatter plots with paths showing temporal progression
through the music universe.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
import plotly.express as px

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

# Set up logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MusicMapVisualizer:
    """Create visualizations of music listening journeys."""

    def __init__(self):
        """Initialize visualizer."""
        pass

    def create_matplotlib_visualization(
        self,
        tracks: List[Dict],
        output_path: Path,
        title: str = "My Music Journey",
        show_labels: bool = False,
        figsize: Tuple[int, int] = config.VIZ_FIGSIZE,
        dpi: int = config.VIZ_DPI
    ):
        """
        Create static visualization using matplotlib.

        Args:
            tracks: List of dicts with 'x', 'y', 'title', 'artist', 'timestamp'
            output_path: Where to save the plot
            title: Plot title
            show_labels: Whether to show track labels
            figsize: Figure size
            dpi: Resolution
        """
        if not tracks:
            logger.error("No tracks to visualize")
            return

        logger.info(f"Creating matplotlib visualization with {len(tracks)} tracks")

        # Extract data
        x_coords = [t['x'] for t in tracks]
        y_coords = [t['y'] for t in tracks]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Create color gradient based on time
        colors = np.linspace(0, 1, len(tracks))
        cmap = plt.cm.get_cmap(config.VIZ_CMAP)

        # Plot path with gradient
        points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=plt.Normalize(0, 1),
            linewidths=2,
            alpha=0.7
        )
        lc.set_array(colors[:-1])
        ax.add_collection(lc)

        # Plot points
        scatter = ax.scatter(
            x_coords,
            y_coords,
            c=colors,
            cmap=cmap,
            s=100,
            alpha=0.8,
            edgecolors='white',
            linewidths=1.5,
            zorder=10
        )

        # Highlight start and end
        ax.scatter(x_coords[0], y_coords[0], s=300, c='green', marker='*',
                  edgecolors='white', linewidths=2, zorder=20, label='Start')
        ax.scatter(x_coords[-1], y_coords[-1], s=300, c='red', marker='*',
                  edgecolors='white', linewidths=2, zorder=20, label='End')

        # Add labels if requested
        if show_labels and len(tracks) <= 20:  # Only for small number of tracks
            for i, track in enumerate(tracks):
                label = f"{track.get('title', 'Unknown')}"
                ax.annotate(
                    label,
                    (x_coords[i], y_coords[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )

        # Styling
        ax.set_xlabel('Music Space Dimension 1', fontsize=12)
        ax.set_ylabel('Music Space Dimension 2', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time (older → recent)', fontsize=10)

        plt.tight_layout()

        # Save
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")

        plt.close()

    def create_plotly_visualization(
        self,
        tracks: List[Dict],
        output_path: Path,
        title: str = "My Music Journey"
    ):
        """
        Create interactive visualization using plotly.

        Args:
            tracks: List of dicts with 'x', 'y', 'title', 'artist', 'timestamp'
            output_path: Where to save the HTML
            title: Plot title
        """
        if not tracks:
            logger.error("No tracks to visualize")
            return

        logger.info(f"Creating plotly visualization with {len(tracks)} tracks")

        # Extract data
        x_coords = [t['x'] for t in tracks]
        y_coords = [t['y'] for t in tracks]
        titles = [t.get('title', 'Unknown') for t in tracks]
        artists = [t.get('artist', 'Unknown') for t in tracks]
        timestamps = [t.get('timestamp', '') for t in tracks]

        # Create hover text
        hover_text = [
            f"<b>{title}</b><br>{artist}<br>{timestamp}"
            for title, artist, timestamp in zip(titles, artists, timestamps)
        ]

        # Create color scale (time-based)
        colors = list(range(len(tracks)))

        # Create figure
        fig = go.Figure()

        # Add path line
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.3)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add points
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time<br>(older → recent)"),
                line=dict(width=1, color='white')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))

        # Highlight start and end
        fig.add_trace(go.Scatter(
            x=[x_coords[0]],
            y=[y_coords[0]],
            mode='markers',
            marker=dict(size=20, color='green', symbol='star', line=dict(width=2, color='white')),
            name='Start',
            hovertext=f"Start: {titles[0]}<br>{artists[0]}",
            hovertemplate='%{hovertext}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=[x_coords[-1]],
            y=[y_coords[-1]],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
            name='End',
            hovertext=f"End: {titles[-1]}<br>{artists[-1]}",
            hovertemplate='%{hovertext}<extra></extra>'
        ))

        # Layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title="Music Space Dimension 1",
            yaxis_title="Music Space Dimension 2",
            hovermode='closest',
            plot_bgcolor='rgba(240, 240, 240, 1)',
            width=1200,
            height=800
        )

        # Save
        output_path.parent.mkdir(exist_ok=True, parents=True)
        fig.write_html(str(output_path))
        logger.info(f"Interactive visualization saved to {output_path}")

    def create_heatmap_visualization(
        self,
        tracks: List[Dict],
        output_path: Path,
        title: str = "Listening Heat Map",
        bins: int = 50
    ):
        """
        Create a heat map showing concentration of listening in different areas.

        Args:
            tracks: List of dicts with 'x', 'y'
            output_path: Where to save the plot
            title: Plot title
            bins: Number of bins for 2D histogram
        """
        if not tracks:
            logger.error("No tracks to visualize")
            return

        logger.info(f"Creating heat map with {len(tracks)} tracks")

        x_coords = [t['x'] for t in tracks]
        y_coords = [t['y'] for t in tracks]

        # Create figure
        fig, ax = plt.subplots(figsize=config.VIZ_FIGSIZE, dpi=config.VIZ_DPI)

        # Create 2D histogram
        h = ax.hist2d(
            x_coords,
            y_coords,
            bins=bins,
            cmap='hot',
            alpha=0.8
        )

        # Styling
        ax.set_xlabel('Music Space Dimension 1', fontsize=12)
        ax.set_ylabel('Music Space Dimension 2', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')

        # Color bar
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Play Count', fontsize=10)

        plt.tight_layout()

        # Save
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=config.VIZ_DPI, bbox_inches='tight')
        logger.info(f"Heat map saved to {output_path}")

        plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize music listening journey")
    parser.add_argument(
        '--tracks-file',
        type=str,
        required=True,
        help='JSON file with track coordinates and metadata'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: auto-generated)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['matplotlib', 'plotly', 'both', 'heatmap'],
        default='both',
        help='Visualization format'
    )
    parser.add_argument(
        '--title',
        type=str,
        default='My Music Journey',
        help='Plot title'
    )
    parser.add_argument(
        '--show-labels',
        action='store_true',
        help='Show track labels (matplotlib only, works best with <20 tracks)'
    )

    args = parser.parse_args()

    # Load tracks
    import json
    with open(args.tracks_file, 'r') as f:
        data = json.load(f)

    tracks = data.get('tracks', [])

    if not tracks:
        logger.error("No tracks found in file")
        return

    logger.info(f"Loaded {len(tracks)} tracks")

    # Create visualizer
    viz = MusicMapVisualizer()

    # Determine output path
    if args.output:
        output_base = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = config.VIZ_OUTPUT_DIR / f"music_journey_{timestamp}"

    # Create visualizations
    if args.format in ['matplotlib', 'both']:
        output_png = output_base.with_suffix('.png')
        viz.create_matplotlib_visualization(
            tracks,
            output_png,
            title=args.title,
            show_labels=args.show_labels
        )

    if args.format in ['plotly', 'both']:
        output_html = output_base.with_suffix('.html')
        viz.create_plotly_visualization(
            tracks,
            output_html,
            title=args.title
        )

    if args.format == 'heatmap':
        output_heatmap = output_base.with_name(f"{output_base.stem}_heatmap.png")
        viz.create_heatmap_visualization(
            tracks,
            output_heatmap,
            title=args.title
        )

    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
