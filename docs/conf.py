# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ASAPY'
copyright = '2024, ASA team'
author = 'ASA team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'autoapi.extension',
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'myst_parser',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'pt-BR'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css', ]


autoapi_dirs = ['../asapy']
#autoapi_file_patterns = ["analysis.py","ml_models.py"]
#autoapi_root = 'api'
autoapi_add_toctree_entry = True
autoapi_keep_files = True


# This is the expected signature of the handler for this event, cf doc


def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    return name.startswith("__init__")


def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member_handler)


master_doc = 'index'

