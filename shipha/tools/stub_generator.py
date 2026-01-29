"""
Stub Generator - Automatically Generate Skeleton Modules from Broken Imports

This tool parses __init__.py files, identifies imports referencing non-existent
modules, and generates skeleton placeholder modules with NotImplementedError stubs.

Usage:
    python -m shipha.tools.stub_generator --package shipha/ --dry-run
    python -m shipha.tools.stub_generator --package shipha/ --force

Features:
    - Recursive scanning of __init__.py files
    - Tree-sitter parsing (v0.25.x) with ast fallback for portability
    - Symbol classification: UPPER_CASE→constant, *Config→dataclass, PascalCase→class
    - Google-style docstrings matching ShiphaEvolve conventions
    - Dry-run mode for preview, --force to overwrite existing files
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# Try tree-sitter first, fallback to ast for portability
try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


@dataclass
class ImportInfo:
    """Represents a single import statement."""

    module: str  # e.g., "shipha.database.sqlite_db"
    names: List[str] = field(default_factory=list)  # e.g., ["ProgramDatabase", "DatabaseConfig"]
    source_file: str = ""  # The __init__.py that contains this import


@dataclass
class SymbolInfo:
    """Represents a symbol to generate."""

    name: str
    symbol_type: str  # "constant", "dataclass", or "class"


@dataclass
class StubModule:
    """Represents a stub module to generate."""

    module_path: str  # e.g., "shipha.database.sqlite_db"
    file_path: Path  # e.g., shipha/database/sqlite_db.py
    symbols: List[SymbolInfo] = field(default_factory=list)
    source_init: str = ""  # The __init__.py that requires this module


class StubGenerator:
    """
    Generates skeleton placeholder modules from broken import statements.

    Uses tree-sitter for parsing when available, falls back to Python's ast module.

    Args:
        package_root: Root directory of the package to scan
        recurse: Whether to recursively scan subdirectories
        verbose: Print detailed progress information
    """

    def __init__(
        self,
        package_root: Path,
        recurse: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the stub generator."""
        self.package_root = Path(package_root).resolve()
        self.recurse = recurse
        self.verbose = verbose
        self.imports: List[ImportInfo] = []
        self.stubs_to_generate: Dict[str, StubModule] = {}

        # Initialize parser
        if TREE_SITTER_AVAILABLE:
            self._py_language = Language(tspython.language())
            self._parser = Parser(self._py_language)
            if self.verbose:
                print("[INFO] Using tree-sitter parser")
        else:
            self._parser = None
            if self.verbose:
                print("[INFO] Using ast fallback parser")

    def scan_package(self) -> List[Path]:
        """
        Find all __init__.py files in the package.

        Returns:
            List of paths to __init__.py files
        """
        init_files: List[Path] = []

        if self.recurse:
            for root, _dirs, files in os.walk(self.package_root):
                if "__init__.py" in files:
                    init_files.append(Path(root) / "__init__.py")
        else:
            top_init = self.package_root / "__init__.py"
            if top_init.exists():
                init_files.append(top_init)

        if self.verbose:
            print(f"[INFO] Found {len(init_files)} __init__.py file(s)")

        return init_files

    def parse_init_file(self, init_path: Path) -> List[ImportInfo]:
        """
        Parse an __init__.py file and extract import statements.

        Args:
            init_path: Path to the __init__.py file

        Returns:
            List of ImportInfo objects
        """
        with open(init_path, "r", encoding="utf-8") as f:
            source = f.read()

        if TREE_SITTER_AVAILABLE:
            return self._parse_with_tree_sitter(source, str(init_path))
        else:
            return self._parse_with_ast(source, str(init_path))

    def _parse_with_tree_sitter(self, source: str, source_file: str) -> List[ImportInfo]:
        """Parse source code using tree-sitter."""
        tree = self._parser.parse(bytes(source, "utf8"))
        root = tree.root_node
        imports: List[ImportInfo] = []

        def traverse(node):
            if node.type == "import_from_statement":
                import_info = self._extract_import_from_ts(node, source, source_file)
                if import_info:
                    imports.append(import_info)

            for child in node.children:
                traverse(child)

        traverse(root)
        return imports

    def _extract_import_from_ts(
        self, node, source: str, source_file: str
    ) -> Optional[ImportInfo]:
        """Extract import info from a tree-sitter import_from_statement node."""
        module_node = node.child_by_field_name("module_name")
        if not module_node:
            return None

        module = source[module_node.start_byte : module_node.end_byte]
        names: List[str] = []

        for child in node.children:
            # Skip the module_name node itself
            if child == module_node:
                continue

            if child.type == "dotted_name":
                name = source[child.start_byte : child.end_byte]
                names.append(name)
            elif child.type == "aliased_import":
                # Get the original name (not alias)
                for subchild in child.children:
                    if subchild.type == "dotted_name":
                        name = source[subchild.start_byte : subchild.end_byte]
                        names.append(name)
                        break

        if names:
            return ImportInfo(module=module, names=names, source_file=source_file)
        return None

    def _parse_with_ast(self, source: str, source_file: str) -> List[ImportInfo]:
        """Parse source code using Python's ast module."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            if self.verbose:
                print(f"[WARN] Syntax error in {source_file}: {e}")
            return []

        imports: List[ImportInfo] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    names = [alias.name for alias in node.names if alias.name != "*"]
                    if names:
                        imports.append(
                            ImportInfo(
                                module=node.module,
                                names=names,
                                source_file=source_file,
                            )
                        )

        return imports

    def check_module_exists(self, module_path: str) -> bool:
        """
        Check if a module file or package exists on disk.

        Args:
            module_path: Dotted module path (e.g., "shipha.database.sqlite_db")

        Returns:
            True if the .py file or package __init__.py exists, False otherwise
        """
        # Convert dotted path to file path relative to package root's parent
        parts = module_path.split(".")
        
        # Check for module file (e.g., shipha/database/sqlite_db.py)
        file_path = self.package_root.parent / Path(*parts[:-1]) / f"{parts[-1]}.py"
        if file_path.exists():
            return True
        
        # Check for package (e.g., shipha/core/__init__.py when importing from shipha.core)
        package_path = self.package_root.parent / Path(*parts) / "__init__.py"
        if package_path.exists():
            return True
        
        return False

    def is_external_module(self, module_path: str) -> bool:
        """
        Check if a module is external (stdlib or third-party).

        Args:
            module_path: Dotted module path

        Returns:
            True if the module is not part of the scanned package
        """
        # Get the package name from package_root
        package_name = self.package_root.name

        # If module doesn't start with package name, it's external
        if not module_path.startswith(package_name + ".") and module_path != package_name:
            return True

        return False

    def classify_symbol(self, name: str) -> str:
        """
        Classify a symbol by its naming convention.

        Args:
            name: Symbol name

        Returns:
            "constant" for UPPER_CASE, "dataclass" for *Config, "class" for PascalCase
        """
        # UPPER_CASE_WITH_UNDERSCORES → constant (string)
        if re.match(r"^[A-Z][A-Z0-9_]*$", name):
            return "constant"

        # Names ending in Config → dataclass
        if name.endswith("Config"):
            return "dataclass"

        # PascalCase → class
        if re.match(r"^[A-Z][a-zA-Z0-9]*$", name):
            return "class"

        # Default to class for anything else
        return "class"

    def collect_stubs_to_generate(self) -> Dict[str, StubModule]:
        """
        Scan package and collect all stub modules that need to be generated.

        Returns:
            Dict mapping module path to StubModule
        """
        init_files = self.scan_package()
        stubs: Dict[str, StubModule] = {}

        for init_path in init_files:
            imports = self.parse_init_file(init_path)

            for imp in imports:
                # Skip external modules (stdlib, third-party)
                if self.is_external_module(imp.module):
                    if self.verbose:
                        print(f"[SKIP] External module: {imp.module}")
                    continue

                # Skip if module exists
                if self.check_module_exists(imp.module):
                    if self.verbose:
                        print(f"[SKIP] Module exists: {imp.module}")
                    continue

                # Convert module path to file path
                parts = imp.module.split(".")
                file_path = self.package_root.parent / Path(*parts[:-1]) / f"{parts[-1]}.py"

                # Create or update stub module
                if imp.module not in stubs:
                    stubs[imp.module] = StubModule(
                        module_path=imp.module,
                        file_path=file_path,
                        symbols=[],
                        source_init=str(init_path),
                    )

                # Add symbols
                for name in imp.names:
                    symbol_type = self.classify_symbol(name)
                    symbol = SymbolInfo(name=name, symbol_type=symbol_type)

                    # Avoid duplicates
                    existing_names = {s.name for s in stubs[imp.module].symbols}
                    if name not in existing_names:
                        stubs[imp.module].symbols.append(symbol)

        self.stubs_to_generate = stubs
        return stubs

    def render_stub_module(self, stub: StubModule) -> str:
        """
        Render a stub module as Python source code.

        Args:
            stub: StubModule to render

        Returns:
            Python source code string
        """
        lines: List[str] = []

        # Module docstring
        module_name = stub.module_path.split(".")[-1]
        lines.append(f'"""')
        lines.append(f"Stub module for {stub.module_path}.")
        lines.append(f"")
        lines.append(f"Auto-generated by shipha.tools.stub_generator.")
        lines.append(f"Pending port from ShinkaEvolve.")
        lines.append(f'"""')
        lines.append("")

        # Imports
        has_dataclass = any(s.symbol_type == "dataclass" for s in stub.symbols)
        has_class = any(s.symbol_type == "class" for s in stub.symbols)

        if has_dataclass:
            lines.append("from dataclasses import dataclass")
        if has_class or has_dataclass:
            lines.append("from typing import Any, Optional")
        lines.append("")

        # Separate by type: constants first, then dataclasses, then classes
        constants = [s for s in stub.symbols if s.symbol_type == "constant"]
        dataclasses = [s for s in stub.symbols if s.symbol_type == "dataclass"]
        classes = [s for s in stub.symbols if s.symbol_type == "class"]

        # Constants
        for sym in constants:
            lines.append(f"# TODO: Port from ShinkaEvolve")
            lines.append(f'{sym.name}: str = ""')
            lines.append(f'"""Placeholder constant. Pending port from ShinkaEvolve."""')
            lines.append("")

        # Dataclasses
        for sym in dataclasses:
            lines.append("")
            lines.append("@dataclass")
            lines.append(f"class {sym.name}:")
            lines.append(f'    """')
            lines.append(f"    Configuration for {self._camel_to_words(sym.name)}.")
            lines.append(f"")
            lines.append(f"    Pending port from ShinkaEvolve.")
            lines.append(f'    """')
            lines.append("")
            lines.append("    pass")
            lines.append("")

        # Classes
        for sym in classes:
            lines.append("")
            lines.append(f"class {sym.name}:")
            lines.append(f'    """')
            lines.append(f"    {self._camel_to_words(sym.name)}.")
            lines.append(f"")
            lines.append(f"    Pending port from ShinkaEvolve.")
            lines.append(f'    """')
            lines.append("")
            lines.append("    def __init__(self, *args: Any, **kwargs: Any) -> None:")
            lines.append(f'        """Initialize {sym.name}."""')
            lines.append(f'        raise NotImplementedError("Pending port from ShinkaEvolve")')
            lines.append("")

        return "\n".join(lines)

    def _camel_to_words(self, name: str) -> str:
        """Convert CamelCase to words."""
        # Insert space before uppercase letters
        words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        # Handle consecutive capitals (e.g., "LLMClient" -> "LLM Client")
        words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", words)
        return words.lower()

    def generate_all(
        self,
        dry_run: bool = False,
        force: bool = False,
    ) -> Tuple[int, int, int]:
        """
        Generate all stub modules.

        Args:
            dry_run: If True, print stubs without writing files
            force: If True, overwrite existing files

        Returns:
            Tuple of (generated, skipped, errors) counts
        """
        stubs = self.collect_stubs_to_generate()

        if not stubs:
            print("[INFO] No stub modules need to be generated.")
            return 0, 0, 0

        generated = 0
        skipped = 0
        errors = 0

        for module_path, stub in stubs.items():
            try:
                # Check if file exists
                if stub.file_path.exists() and not force:
                    if self.verbose or dry_run:
                        print(f"[SKIP] {stub.file_path} already exists (use --force to overwrite)")
                    skipped += 1
                    continue

                # Render the stub
                source = self.render_stub_module(stub)

                if dry_run:
                    print(f"\n{'=' * 60}")
                    print(f"[DRY-RUN] Would create: {stub.file_path}")
                    print(f"{'=' * 60}")
                    print(source)
                    generated += 1
                else:
                    # Ensure parent directory exists
                    stub.file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write the file
                    with open(stub.file_path, "w", encoding="utf-8") as f:
                        f.write(source)

                    print(f"[CREATE] {stub.file_path}")
                    generated += 1

            except Exception as e:
                print(f"[ERROR] Failed to generate {module_path}: {e}")
                errors += 1

        return generated, skipped, errors


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point for stub generator.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        description="Generate skeleton placeholder modules from broken imports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview what would be generated
    python -m shipha.tools.stub_generator --package shipha/ --dry-run

    # Generate stubs (skip existing files)
    python -m shipha.tools.stub_generator --package shipha/

    # Force overwrite existing stubs
    python -m shipha.tools.stub_generator --package shipha/ --force

    # Scan only top-level __init__.py
    python -m shipha.tools.stub_generator --package shipha/ --no-recurse
        """,
    )

    parser.add_argument(
        "--package",
        "-p",
        type=Path,
        default=Path("."),
        help="Root package directory to scan (default: current directory)",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Print generated stubs without writing files",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing stub files",
    )

    parser.add_argument(
        "--no-recurse",
        action="store_true",
        help="Only scan top-level __init__.py (default: recursive)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args(argv)

    # Validate package path
    if not args.package.exists():
        print(f"[ERROR] Package path does not exist: {args.package}")
        return 1

    if not args.package.is_dir():
        print(f"[ERROR] Package path is not a directory: {args.package}")
        return 1

    # Create generator
    generator = StubGenerator(
        package_root=args.package,
        recurse=not args.no_recurse,
        verbose=args.verbose,
    )

    # Run generation
    generated, skipped, errors = generator.generate_all(
        dry_run=args.dry_run,
        force=args.force,
    )

    # Summary
    print("")
    print(f"[SUMMARY] Generated: {generated}, Skipped: {skipped}, Errors: {errors}")

    if TREE_SITTER_AVAILABLE:
        print("[INFO] Parser: tree-sitter")
    else:
        print("[INFO] Parser: ast (tree-sitter not available)")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
