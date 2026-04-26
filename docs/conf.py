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
from pathlib import Path

from pygments.formatters import LatexFormatter

project = "Qiskit-Braket provider"
copyright = f"{datetime.datetime.now(tz=datetime.UTC).year}, Amazon.com"  # noqa: A001
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
