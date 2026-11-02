# illustrations.py -*- Python -*-
#
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""MkDocs hook: show a page's generated illustration as one tabbed viewer.

On GitHub, an illustration is a ``<picture>`` per view between
``<!-- illustration: NAME ... -->`` and ``<!-- /illustration -->``: the
first view inline, the deeper ones folded in ``<details>``. GitHub matches
each ``<picture>``'s source to the reader's GitHub theme.

On the site neither works well: Material's light/dark toggle sets
``data-md-color-scheme``, which a ``<picture>`` cannot see, and the views
read better side by side than folded. This hook replaces the block with
tabs, one per view, each holding both themes' images;
``docs/assets/css/iron.css`` shows the one matching the toggle and
``docs/assets/javascripts/illustrations.js`` switches tabs. The images keep
the paths the markdown gives, so they resolve as they do on GitHub.
"""

import html
import re

_BLOCK = re.compile(
    r"<!-- illustration: (?P<name>[\w-]+)[^>]*-->(?P<body>.*?)<!-- /illustration -->",
    re.S,
)
_PICTURE = re.compile(
    r'<source media="\(prefers-color-scheme: dark\)" srcset="(?P<dark>[^"]+)">\s*'
    r'<img src="(?P<light>[^"]+)"[^>]*?alt="(?P<alt>[^"]*)"',
    re.S,
)
# assets/NAME.VIEW.light.svg
_VIEW = re.compile(r"\.(?P<view>[\w-]+)\.light\.svg$")


def _viewer(name: str, body: str) -> str:
    views = []
    for m in _PICTURE.finditer(body):
        v = _VIEW.search(m["light"])
        views.append(
            (v["view"] if v else str(len(views)), m["light"], m["dark"], m["alt"])
        )
    if not views:
        return body
    tabs, panes = [], []
    for i, (view, light, dark, alt) in enumerate(views):
        sel = "true" if i == 0 else "false"
        tabs.append(
            f'<button role="tab" aria-selected="{sel}" data-view="{view}">'
            f"{html.escape(view.capitalize())}</button>"
        )
        hidden = "" if i == 0 else " hidden"
        panes.append(
            f'<div class="iron-ill-pane" data-view="{view}"{hidden}>'
            f'<img class="iron-ill-light" src="{light}" alt="{alt}">'
            f'<img class="iron-ill-dark" src="{dark}" alt="{alt}"></div>'
        )
    return (
        f'<div class="iron-ill" id="{name}">'
        f'<div class="iron-ill-tabs" role="tablist">{"".join(tabs)}</div>'
        f'{"".join(panes)}</div>'
    )


def on_page_markdown(markdown, page, config, files):
    return _BLOCK.sub(lambda m: _viewer(m["name"], m["body"]), markdown)
