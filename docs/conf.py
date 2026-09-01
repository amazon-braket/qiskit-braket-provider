# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Portions Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Sphinx documentation builder
"""

# General options:
import datetime
import os
from pathlib import Path

from pygments.formatters import LatexFormatter
from sphinx.application import Sphinx

project = "Qiskit-Braket provider"
copyright = f"{datetime.datetime.now(tz=datetime.UTC).year}, Amazon.com"  # ruff:ignore[builtin-variable-shadowing]
author = "Amazon Web Services"

# The full version, including alpha/beta/rc tags
with (Path(__file__).resolve().parent / ".." / "qiskit_braket_provider" / "_version.py").open(
    encoding="utf-8"
) as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")
release = version

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "qiskit_sphinx_theme",
    "sphinxcontrib.rsvgconverter",
]
templates_path = ["_templates"]
numfig = True
numfig_format = {"table": "Table %s"}
language = "en"
pygments_style = "colorful"
add_module_names = False
modindex_common_prefix = ["qiskit_braket_provider."]

# html theme options
html_static_path = ["_static"]
html_logo = "_static/images/logo.png"

# autodoc/autosummary options
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"

# nbsphinx options (for tutorials)
nbsphinx_timeout = 180
nbsphinx_execute = "never"
nbsphinx_widgets_path = ""
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"

# LaTeX options (for PDF builds on Read the Docs)
latex_engine = "lualatex"
latex_elements = {
    "preamble": rf"""
\providecommand{{\mathbfit}}[1]{{\boldsymbol{{#1}}}}
{LatexFormatter().get_style_defs()}
""",
}

LLMS_TXT_TITLE = "Qiskit-Braket Provider"
LLMS_TXT_SUMMARY = (
    "Provider that runs Qiskit programs on quantum computing devices and "
    "simulators through Amazon Braket."
)
LLMS_TXT_BASE_URL = "https://qiskit-braket-provider.readthedocs.io/en/stable/"
LLMS_TXT_SECTIONS: dict[str, tuple[str, ...]] = {
    "Docs": (),
    "Examples": ("tutorials/", "how_tos/"),
    # autosummary_generate writes the per-object stub pages into stubs/.
    "API Reference": ("apidocs/", "stubs/"),
}


def _llms_txt_section(docname: str) -> str:
    """Return the llms.txt section heading a document belongs under.

    Sections are tried in declaration order, so the first matching prefix wins.
    A document that matches no prefix goes under the first section.
    """
    for heading, prefixes in LLMS_TXT_SECTIONS.items():
        if any(docname.startswith(prefix) for prefix in prefixes):
            return heading
    default_heading, _ = next(iter(LLMS_TXT_SECTIONS.items()))
    return default_heading


def _write_llms_txt(app: Sphinx, exception: Exception | None) -> None:
    """Write llms.txt, a manifest of every built page for LLM discoverability.

    The format follows https://llmstxt.org: an H1 name, a blockquote summary, then
    one file list per H2 section. Pages are grouped so that an agent can tell
    narrative docs, runnable examples and generated API reference apart.
    """
    if exception or app.builder.name != "html":
        return

    # Read the Docs passes the canonical URL to every build automatically, so this
    # is set in any RTD build and the default only applies elsewhere. See
    # https://docs.readthedocs.com/platform/stable/canonical-urls.html#how-to-specify-the-canonical-url
    base_url = os.environ.get("READTHEDOCS_CANONICAL_URL", LLMS_TXT_BASE_URL)
    if base_url and not base_url.endswith("/"):
        base_url += "/"

    env = app.env
    sections: dict[str, list[str]] = {heading: [] for heading in LLMS_TXT_SECTIONS}
    for docname in sorted(env.all_docs):
        url = f"{base_url}{app.builder.get_target_uri(docname)}"
        sections[_llms_txt_section(docname)].append(f"- [{env.titles[docname].astext()}]({url})")

    lines = [f"# {LLMS_TXT_TITLE}", "", f"> {LLMS_TXT_SUMMARY}"]
    for heading in LLMS_TXT_SECTIONS:
        if sections[heading]:
            lines += ["", f"## {heading}", "", *sections[heading]]

    out = Path(app.outdir) / "llms.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"--> Wrote {out.name}")


def setup(app: Sphinx) -> None:
    """Register build hooks."""
    app.connect("build-finished", _write_llms_txt)
