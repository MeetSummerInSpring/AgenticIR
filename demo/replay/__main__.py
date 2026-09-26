from __future__ import annotations

import argparse
from pathlib import Path

from .builder import ReplayBuilder
from .server import WEB_ROOT, serve


DEFAULT_OUTPUT = WEB_ROOT / "data" / "current"


def build_command(args: argparse.Namespace) -> None:
    manifest = ReplayBuilder(
        run_dir=args.run_dir,
        output_dir=args.output,
        title=args.title,
    ).build()
    print(f"回放数据已生成：{manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgenticIR 历史运行回放演示")
    commands = parser.add_subparsers(dest="command", required=True)

    build_parser = commands.add_parser("build", help="从历史运行生成回放数据")
    build_parser.add_argument("run_dir", type=Path, help="一次完整的 output/<run> 目录")
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build_parser.add_argument("--title", default=None, help="演示标题")
    build_parser.set_defaults(handler=build_command)

    serve_parser = commands.add_parser("serve", help="启动本地演示服务")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(handler=lambda args: serve(args.host, args.port))

    all_parser = commands.add_parser("all", help="生成回放数据并启动服务")
    all_parser.add_argument("run_dir", type=Path)
    all_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    all_parser.add_argument("--title", default=None)
    all_parser.add_argument("--host", default="127.0.0.1")
    all_parser.add_argument("--port", type=int, default=8000)

    def build_and_serve(args: argparse.Namespace) -> None:
        build_command(args)
        serve(args.host, args.port)

    all_parser.set_defaults(handler=build_and_serve)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
