import argparse
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.application import RouteMapApplication
from app.core.config import AppConfig


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Interactive Route Map Application")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()

    # Initialize configuration
    config = AppConfig.from_file(args.config) if args.config else AppConfig()

    # Initialize and run application
    app = RouteMapApplication(config)
    app.run()


if __name__ == "__main__":
    main()
