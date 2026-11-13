/* Clean Voice's download page: picks the button for the visitor's computer and builds every GitHub link from the
   page's own address. No libraries, no cookies, nothing is sent anywhere. Without this script both buttons show and
   the links in the HTML (the fallback repo below) still work. Tested by tests/test_download_page.py. */
(function (root) {
  "use strict";

  // FALLBACK: used when the page is not served from <owner>.github.io/<repo>/ (a local file, a preview server).
  // It is the project's own repo, pyproject.toml's Homepage; the HTML's default links use the same one.
  var FALLBACK_REPO = "paulcaesarflores/clean-voice";

  var FILES = { windows: "Clean-Voice-Windows.msi", mac: "Clean-Voice-macOS.dmg" };

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

  /* Sets every [data-link] address, every [data-cable] name, the [data-shot] screenshot and the page's data-os,
     which the style sheet reads. */
  function apply(doc, nav, loc) {
    var os = detectOS(nav.userAgentData, nav.userAgent, nav.maxTouchPoints);
    var urls = links(repoFromLocation(loc.hostname, loc.pathname));
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
    doc.documentElement.setAttribute("data-os", os);
  }

  root.CleanVoiceDownload = {
    FALLBACK_REPO: FALLBACK_REPO,
    FILES: FILES,
    CABLE_MIC: CABLE_MIC,
    SHOTS: SHOTS,
    detectOS: detectOS,
    repoFromLocation: repoFromLocation,
    links: links,
    apply: apply
  };

  if (typeof document !== "undefined") {
    apply(document, navigator, location);
  }
})(this);
