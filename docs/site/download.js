/* Clean Voice's download page: builds every GitHub link and the install commands from the page's own address, wires the
   Copy buttons, and names the call-app microphone and the screenshot for the visitor's computer. The Windows and Mac tabs show one
   computer's install steps at a time (the visitor's own first); without this script both show. No libraries, no cookies, nothing is sent anywhere. Without this script the links and commands in the
   HTML (the fallback repo below) still work. Tested by tests/test_download_page.py. */
(function (root) {
  "use strict";

  // FALLBACK: used when the page is not served from <owner>.github.io/<repo>/ (a local file, a preview server).
  // It is the project's own repo, pyproject.toml's Homepage; the HTML's default links use the same one.
  var FALLBACK_REPO = "paulcaesarflores/clean-voice";

  var FILES = { windows: "Clean-Voice-Windows.msi", mac: "Clean-Voice-macOS.dmg" };

  // The release's install scripts (tools/ci/release_assets.py SCRIPT_NAMES): the command fetches the one for the OS.
  var SCRIPTS = { windows: "install.ps1", mac: "install.sh" };

  // Read from the user agent text when the browser has no userAgentData. First match wins: phones and tablets come
  // first, because iPhone user agents say "like Mac OS X" and Android ones say "Linux".
  var UA_RULES = [
    [/iPhone|iPad|iPod|Android|Mobile/i, "other"],
    [/Windows/i, "windows"],
    [/Macintosh|Mac OS X/i, "mac"]
  ];

  // The microphone to pick in the call app: the cable's recording side (src/clean_voice/app.py cable_mic_name). The
  // "other" text is also the HTML's own, for visitors without JavaScript.
  var CABLE_MIC = {
    windows: "“CABLE Output”",
    mac: "“VB-Cable”",
    other: "“CABLE Output” on Windows, “VB-Cable” on Mac,"
  };

  // The Home screenshot, light and dark, rendered with that computer's text (tools/ui_snapshot.py --platform). Every
  // visitor but a Mac one keeps the HTML's own, the Windows one.
  var SHOTS = {
    windows: { light: "running.png", dark: "running-dark.png" },
    mac: { light: "running-mac.png", dark: "running-mac-dark.png" }
  };

  // userAgentData.platform values (Chromium browsers).
  var PLATFORMS = { Windows: "windows", macOS: "mac" };

  /* "windows", "mac" or "other" (Linux, phones, tablets, anything unknown). */
  function detectOS(uaData, userAgent, maxTouchPoints) {
    var os = "other";
    if (uaData && uaData.platform) {
      os = uaData.mobile ? "other" : PLATFORMS[uaData.platform] || "other";
    } else {
      var ua = userAgent || "";
      for (var i = 0; i < UA_RULES.length; i++) {
        if (UA_RULES[i][0].test(ua)) {
          os = UA_RULES[i][1];
          break;
        }
      }
    }
    // An iPad asks for the desktop site with a Mac user agent; only its touch screen gives it away.
    if (os === "mac" && maxTouchPoints > 1) os = "other";
    return os;
  }

  /* "owner/repo" from a GitHub Pages address (<owner>.github.io/<repo>/), else the fallback. */
  function repoFromLocation(hostname, pathname) {
    var suffix = ".github.io";
    var host = (hostname || "").toLowerCase();
    if (host.length > suffix.length && host.slice(-suffix.length) === suffix) {
      var owner = host.slice(0, -suffix.length);
      var repo = (pathname || "").split("/").filter(Boolean)[0];
      if (repo) return owner + "/" + repo;
    }
    return FALLBACK_REPO;
  }

  function links(repo) {
    var base = "https://github.com/" + repo;
    return {
      windows: base + "/releases/latest/download/" + FILES.windows,
      mac: base + "/releases/latest/download/" + FILES.mac,
      source: base,
      licence: base + "/blob/main/LICENSE"
    };
  }

  /* The command to paste: PowerShell's irm | iex on Windows, curl | bash on a Mac. The same text as
     clean_voice.updates.install_command_text (tests/test_download_page.py), which Install update runs. */
  function commands(repo) {
    var base = "https://github.com/" + repo + "/releases/latest/download/";
    return {
      windows: "irm " + base + SCRIPTS.windows + " | iex",
      mac: "curl -fsSL " + base + SCRIPTS.mac + " | bash"
    };
  }

  /* Shows one computer's panel and marks its tab (the tabs pattern of the WAI-ARIA practices: aria-selected, one tab stop). */
  function selectTab(doc, name) {
    var tabs = doc.querySelectorAll("[data-tab]");
    for (var i = 0; i < tabs.length; i++) {
      var on = tabs[i].getAttribute("data-tab") === name;
      tabs[i].setAttribute("aria-selected", on ? "true" : "false");
      tabs[i].setAttribute("tabindex", on ? "0" : "-1");
    }
    var panels = doc.querySelectorAll("[data-os-target]");
    for (var j = 0; j < panels.length; j++) {
      panels[j].hidden = panels[j].getAttribute("data-os-target") !== name;
    }
  }

  /* Sets every [data-link] address, every [data-command] text, every [data-cable] name, the [data-shot] screenshot and the page's data-os,
     which the style sheet reads (it shows the Copy buttons). */
  function apply(doc, nav, loc) {
    var os = detectOS(nav.userAgentData, nav.userAgent, nav.maxTouchPoints);
    var repo = repoFromLocation(loc.hostname, loc.pathname);
    var urls = links(repo);
    var texts = commands(repo);
    var boxes = doc.querySelectorAll("[data-command]");
    for (var c = 0; c < boxes.length; c++) {
      var text = texts[boxes[c].getAttribute("data-command")];
      if (text) boxes[c].textContent = text;
    }
    var anchors = doc.querySelectorAll("[data-link]");
    for (var i = 0; i < anchors.length; i++) {
      var url = urls[anchors[i].getAttribute("data-link")];
      if (url) anchors[i].setAttribute("href", url);
    }
    var cables = doc.querySelectorAll("[data-cable]");
    for (var j = 0; j < cables.length; j++) {
      cables[j].textContent = CABLE_MIC[os];
    }
    var shots = doc.querySelectorAll("[data-shot]");
    for (var k = 0; k < shots.length; k++) {
      var shot = (os === "mac" ? SHOTS.mac : SHOTS.windows)[shots[k].getAttribute("data-shot")];
      if (shot) shots[k].setAttribute(shots[k].tagName.toLowerCase() === "source" ? "srcset" : "src", shot);
    }
    selectTab(doc, os === "mac" ? "mac" : "windows");
    doc.documentElement.setAttribute("data-os", os);
  }

  /* The Copy button: puts the command on the clipboard and says "Copied" for a moment. Without a clipboard API (an old
     browser, a local file) it selects the command so Ctrl+C or Cmd+C finishes the job. */
  function copyCommand(nav, text, done, fail) {
    if (nav.clipboard && nav.clipboard.writeText) {
      nav.clipboard.writeText(text).then(done, fail);
    } else {
      fail();
    }
  }

  /* Wires the tabs and the Copy buttons (a real page only). */
  function wire(doc, nav, loc) {
    var copy = function (button) {
      button.addEventListener("click", function () {
        var box = doc.querySelector('[data-command="' + button.getAttribute("data-copy") + '"]');
        var label = button.textContent;
        var restore = function () { button.textContent = label; };
        copyCommand(nav, box.textContent, function () {
          button.textContent = "Copied";
          setTimeout(restore, 2000);
        }, function () {
          var range = doc.createRange();
          range.selectNodeContents(box);
          var selection = doc.defaultView.getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
          button.textContent = "Press Ctrl+C";
          setTimeout(restore, 3000);
        });
      });
    };
    var tabs = doc.querySelectorAll("[data-tab]");
    var pick = function (index) {
      var tab = tabs[(index + tabs.length) % tabs.length];
      selectTab(doc, tab.getAttribute("data-tab"));
      tab.focus();
    };
    var tab = function (index) {
      tabs[index].addEventListener("click", function () { selectTab(doc, tabs[index].getAttribute("data-tab")); });
      tabs[index].addEventListener("keydown", function (event) {
        var keys = { ArrowLeft: index - 1, ArrowUp: index - 1, ArrowRight: index + 1, ArrowDown: index + 1, Home: 0, End: tabs.length - 1 };
        if (event.key in keys) {
          event.preventDefault();
          pick(keys[event.key]);
        }
      });
    };
    for (var t = 0; t < tabs.length; t++) tab(t);
    var copies = doc.querySelectorAll("[data-copy]");
    for (var i = 0; i < copies.length; i++) copy(copies[i]);
  }

  root.CleanVoiceDownload = {
    FALLBACK_REPO: FALLBACK_REPO,
    FILES: FILES,
    SCRIPTS: SCRIPTS,
    CABLE_MIC: CABLE_MIC,
    SHOTS: SHOTS,
    detectOS: detectOS,
    repoFromLocation: repoFromLocation,
    links: links,
    commands: commands,
    copyCommand: copyCommand,
    selectTab: selectTab,
    apply: apply
  };

  if (typeof document !== "undefined") {
    apply(document, navigator, location);
    wire(document, navigator, location);
  }
})(this);
