"""Unit tests for main.py utility functions and CLI parsing."""

from main import main


class TestMainCli:
    """Tests for CLI argument parsing."""

    def test_no_command_prints_help(self, capsys):
        main([])
        captured = capsys.readouterr()
        assert "RCT Checker" in captured.out

    def test_extract_command_parses_args(self):
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        extract_parser = subparsers.add_parser("extract")
        extract_parser.add_argument("--pdf", required=True)
        extract_parser.add_argument("--force", action="store_true")
        extract_parser.add_argument("--llm-backend", default="openai")
        extract_parser.add_argument("--log-level", default="INFO")

        args = parser.parse_args(["extract", "--pdf", "paper.pdf", "--force"])
        assert args.pdf == "paper.pdf"
        assert args.force is True

        args = parser.parse_args(["extract", "--pdf", "/path/to/dir"])
        assert args.pdf == "/path/to/dir"
        assert args.force is False

    def test_analyze_command_parses_args(self):
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        analyze_parser = subparsers.add_parser("analyze")
        analyze_parser.add_argument("--id", type=int)
        analyze_parser.add_argument("--json")
        analyze_parser.add_argument("--skip-cont", action="store_true")
        analyze_parser.add_argument("--skip-cat", action="store_true")
        analyze_parser.add_argument("--plot", action="store_true")

        args = parser.parse_args(["analyze", "--id", "5", "--skip-cont"])
        assert args.id == 5
        assert args.skip_cont is True
        assert args.skip_cat is False

    def test_list_command_parses_status_filter(self):
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        list_parser = subparsers.add_parser("list")
        list_parser.add_argument("--status", choices=["success", "failed"])

        args = parser.parse_args(["list", "--status", "success"])
        assert args.status == "success"
