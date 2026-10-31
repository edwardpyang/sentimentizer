import argparse
import os
import sys

_DIFFUSION_MOVED_MESSAGE = (
    "image generation moved to the imagenate service; see docs/serving.md. "
    "Remove --diffusion and SENTIMENTIZER_DIFFUSION_ENABLED."
)


def _positive_int(value: str) -> int:
    """Return a positive integer for argparse."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the serve command-line parser."""
    parser = argparse.ArgumentParser(description="Start Sentimentizer Serve")
    parser.add_argument(
        "--host",
        default=None,
        help="Override the serve_host configured in service.yaml",
    )
    parser.add_argument(
        "--port",
        type=_positive_int,
        default=None,
        help="Override the serve_port configured in service.yaml",
    )
    parser.add_argument(
        "--ray-object-store-memory-mb",
        type=_positive_int,
        default=None,
        help="Override ray_object_store_memory_mb from service.yaml",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to service.yaml (default: bundled config, "
        "or SENTIMENTIZER_SERVE_CONFIG env var). "
        "Must be set before module imports, so it's promoted to "
        "SENTIMENTIZER_SERVE_CONFIG.",
    )
    env_diffusion = os.environ.get("SENTIMENTIZER_DIFFUSION_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
    )
    env_embeddings = os.environ.get("SENTIMENTIZER_EMBEDDINGS_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
    )
    parser.add_argument(
        "--diffusion",
        action="store_true",
        default=env_diffusion,
        help="Removed. Exits with an error: image generation moved to the imagenate service.",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        default=env_embeddings,
        help="Enable dense embedding routes. BGE-M3 additionally requires "
        "SENTIMENTIZER_BGE_M3_ENABLED=1.",
    )
    parser.add_argument(
        "--bge-m3-only",
        action="store_true",
        help="Serve only BGE-M3 embeddings and health routes.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Start the requested Serve application."""
    if args.diffusion:
        print(_DIFFUSION_MOVED_MESSAGE, file=sys.stderr)
        sys.exit(2)

    if args.config:
        os.environ["SENTIMENTIZER_SERVE_CONFIG"] = args.config
    if args.bge_m3_only:
        os.environ["SENTIMENTIZER_EMBEDDINGS_ENABLED"] = "1"
        os.environ["SENTIMENTIZER_BGE_M3_ENABLED"] = "1"
    if args.embeddings:
        os.environ["SENTIMENTIZER_EMBEDDINGS_ENABLED"] = "1"

    from sentimentizer.serve.config import load_serve_config

    serve_cfg = load_serve_config(args.config)
    host = args.host if args.host is not None else serve_cfg.serve_host
    port = args.port if args.port is not None else serve_cfg.serve_port
    object_store_memory_mb = (
        args.ray_object_store_memory_mb
        if args.ray_object_store_memory_mb is not None
        else serve_cfg.ray_object_store_memory_mb
    )

    if args.bge_m3_only:
        from sentimentizer.serve.bge_only_app import main

        main(
            host=host,
            port=port,
            object_store_memory_mb=object_store_memory_mb,
        )
        return

    from sentimentizer.serve.app import main

    main(host=host, port=port)


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
