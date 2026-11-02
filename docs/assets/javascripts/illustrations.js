// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Tabs for the illustration viewers docs/hooks/illustrations.py writes.
document$.subscribe(function () {
  document.querySelectorAll(".iron-ill").forEach(function (viewer) {
    var tabs = viewer.querySelectorAll(".iron-ill-tabs button");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        var view = tab.dataset.view;
        tabs.forEach(function (t) {
          t.setAttribute("aria-selected", t === tab ? "true" : "false");
        });
        viewer.querySelectorAll(".iron-ill-pane").forEach(function (pane) {
          pane.hidden = pane.dataset.view !== view;
        });
      });
    });
  });
});
