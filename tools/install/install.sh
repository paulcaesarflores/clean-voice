#!/bin/bash
# Clean Voice installer for macOS 13 or later. Public, MIT, ASCII only.
#
#   curl -fsSL https://github.com/paulcaesarflores/clean-voice/releases/latest/download/install.sh | bash
#
# It gets its own Python (uv's managed one), fetches this release's source, checks it, builds Clean Voice with
# Briefcase (create + build) and installs it in Applications, replacing a copy installed from the old DMG.
# Runs under macOS's own bash 3.2: no associative arrays, no ${var,,}.
#
# Rules kept here: one main function called on the last line (a half-downloaded pipe runs nothing); every child
# program reads </dev/null (so none of them eats the script from the pipe). The tests source this file with
# CLEAN_VOICE_TEST_IMPORT set, so it only defines its functions.

# Source: uv's release page (.sha256 files); Briefcase pin equals pyproject.toml's dev group. Stamped by
# release_assets.py: the version and the source zip's SHA-256.
VERSION='@@VERSION@@'
SOURCE_SHA256='@@SOURCE_SHA256@@'
BRIEFCASE_PIN='0.4.7'
UV_VERSION='0.13.9'
UV_SHA256_ARM64='5aeca67f2283f75235de40d0bd9c0a5bbc5fce23b6d48b3bf9fa1b2db08382ca'
UV_SHA256_X86_64='617a3211261b0fc739562f0071aa34e9739797c510208ede8e2977c30903d041'
DEFAULT_BASE='https://github.com/paulcaesarflores/clean-voice/releases/latest/download'
APP_NAME='Clean Voice.app'

# Settings, set by init_vars; the tests change them after sourcing this file.
# CLEAN_VOICE_BASE and CLEAN_VOICE_TEST_IMPORT are the two test hooks; CLEAN_VOICE_UPDATE, _WAIT_PID and _STATUS_FILE
# are set by the app's Install update (src/clean_voice/updates.py).
init_vars() {
    BASE="${CLEAN_VOICE_BASE:-$DEFAULT_BASE}"
    BASE="${BASE%/}"
    WORK="$HOME/Library/Caches/Clean Voice Setup"
    LOG="$HOME/Library/Logs/Clean Voice Install.log"
    APPS_SYSTEM='/Applications'
    APPS_USER="$HOME/Applications"
    UPDATE="${CLEAN_VOICE_UPDATE:-}"
    WAIT_PID="${CLEAN_VOICE_WAIT_PID:-}"
    STATUS_FILE="${CLEAN_VOICE_STATUS_FILE:-}"
    POLL_INTERVAL=10
    POLL_TRIES=270
    WAIT_SLEEP=0.2
    WAIT_TRIES=300
    STEP=0
    STEP_NAME=''
}

# ---- Output, log, errors -------------------------------------------------------------------------------------

say() {
    printf '%s\n' "$1"
    log "$1"
}

log() {
    if [ -n "${LOG:-}" ]; then
        mkdir -p "$(dirname "$LOG")" 2>/dev/null
        printf '%s\n' "$1" >>"$LOG" 2>/dev/null
    fi
}

step() {
    STEP="$1"
    STEP_NAME="$2"
    say "Step $1 of 7: $2"
}

# The one-word status file Install update reads once a second ("swap" or "failed"); no OBS source: the user approved it.
write_status() {
    if [ -n "$STATUS_FILE" ]; then
        printf '%s' "$1" >"$STATUS_FILE" 2>/dev/null
    fi
}

# A plain failure: print it, report "failed" to the app, stop. The EXIT trap removes the setup folder.
die() {
    say "$1"
    write_status failed
    exit 1
}

die_step() {
    log "Failed: $1"
    die "Something went wrong at step $STEP ($STEP_NAME). Nothing was changed. The log is at $LOG; send it to your host."
}

# Run a program quietly, its output to the log, stdin from /dev/null.
run_logged() {
    log "> $*"
    "$@" </dev/null >>"$LOG" 2>&1
}

# ---- Checks ----------------------------------------------------------------------------------------------------

# Source: pyproject.toml min_os_version = "13.0".
check_macos() {
    local major="${1%%.*}"
    case "$major" in
        '' | *[!0-9]*) die 'Clean Voice needs macOS 13 or later.' ;;
    esac
    if [ "$major" -lt 13 ]; then
        die 'Clean Voice needs macOS 13 or later.'
    fi
}

app_running() {
    pgrep -x 'Clean Voice' >/dev/null 2>&1
}

# Source: uname -m picks uv's archive (arm64 or x86_64).
uv_archive() {
    case "$(uname -m)" in
        arm64) echo 'uv-aarch64-apple-darwin' ;;
        x86_64) echo 'uv-x86_64-apple-darwin' ;;
        *) return 1 ;;
    esac
}

uv_sha256() {
    case "$1" in
        uv-aarch64-apple-darwin) echo "$UV_SHA256_ARM64" ;;
        *) echo "$UV_SHA256_X86_64" ;;
    esac
}

# Source: briefcase/platforms/macOS/xcode.py:220-247 (create and build need the Command Line Tools); Apple's
# xcode-select: --install opens Apple's installer window, so we wait for it.
ensure_tools() {
    if xcode-select -p >/dev/null 2>&1; then
        return 0
    fi
    xcode-select --install >/dev/null 2>&1
    say "Apple is asking to install its developer tools. Click Install and wait; Clean Voice continues by itself."
    local tries=0
    while [ "$tries" -lt "$POLL_TRIES" ]; do
        if xcode-select -p >/dev/null 2>&1; then
            return 0
        fi
        tries=$((tries + 1))
        sleep "$POLL_INTERVAL"
    done
    die "Run the same command again once Apple's tools have finished installing."
}

# ---- Downloads -------------------------------------------------------------------------------------------------

fetch() {
    log "Downloading $1"
    if ! curl -fsSL -o "$2" "$1" </dev/null 2>>"$LOG"; then
        die "Couldn't reach GitHub. Check your internet connection and run the command again."
    fi
}

check_sha256() {
    local actual
    actual="$(shasum -a 256 "$1" | cut -d ' ' -f 1)"
    if [ "$actual" != "$2" ]; then
        die "The download didn't match what was published, so nothing was installed."
    fi
}

# ---- Install into Applications -----------------------------------------------------------------------------------

# Where the app goes: an existing copy is replaced where it is; otherwise /Applications if the guest can write there,
# else ~/Applications.
pick_target() {
    if [ -e "$APPS_SYSTEM/$APP_NAME" ]; then
        echo "$APPS_SYSTEM/$APP_NAME"
    elif [ -e "$APPS_USER/$APP_NAME" ]; then
        echo "$APPS_USER/$APP_NAME"
    elif [ -w "$APPS_SYSTEM" ]; then
        echo "$APPS_SYSTEM/$APP_NAME"
    else
        mkdir -p "$APPS_USER"
        echo "$APPS_USER/$APP_NAME"
    fi
}

wait_for_app_exit() {
    if [ -z "$WAIT_PID" ]; then
        return 0
    fi
    local tries=0
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        tries=$((tries + 1))
        if [ "$tries" -gt "$WAIT_TRIES" ]; then
            return 1
        fi
        sleep "$WAIT_SLEEP"
    done
}

# Source: Apple's ditto (copies a bundle with its signature and attributes) and open. The new copy is made beside
# the old one, the old one is renamed aside, and any failure puts it back (rollback only). In update mode "swap" is
# written first and the old app is reopened after a failed swap.
swap_app() {
    local new="$1" target="$2" had_old=0
    [ -e "$target" ] && had_old=1
    local unreplaceable="The Clean Voice in Applications can't be replaced from your account. Drag it to the Bin, then run the command again."

    rm -rf "$target.new"
    if ! ditto "$new" "$target.new" </dev/null 2>>"$LOG"; then
        rm -rf "$target.new"
        if [ "$had_old" = 1 ] && [ -z "$UPDATE" ]; then die "$unreplaceable"; fi
        die_step 'could not copy the new app'
    fi
    if [ -n "$UPDATE" ]; then
        write_status swap
        if ! wait_for_app_exit; then
            rm -rf "$target.new"
            die "Clean Voice didn't close, so the old version is still in place."
        fi
    fi
    rm -rf "$target.old"
    if [ "$had_old" = 1 ] && ! mv "$target" "$target.old" 2>>"$LOG"; then
        rm -rf "$target.new"
        undo_swap "$target" "$had_old" "$unreplaceable"
    fi
    if ! mv "$target.new" "$target" 2>>"$LOG"; then
        undo_swap "$target" "$had_old" "$unreplaceable"
    fi
    rm -rf "$target.old"
}

undo_swap() {
    local target="$1" had_old="$2" unreplaceable="$3"
    rm -rf "$target.new"
    if [ -e "$target.old" ]; then
        rm -rf "$target"
        mv "$target.old" "$target" 2>>"$LOG"
    fi
    if [ "$had_old" = 1 ] && [ -n "$UPDATE" ]; then
        open "$target" </dev/null >/dev/null 2>&1
        die "Couldn't replace the old Clean Voice. Your old version is back and was reopened."
    fi
    if [ "$had_old" = 1 ]; then die "$unreplaceable"; fi
    die_step 'could not move the new app into place'
}

# The marker the app reads to know the command installed it, so Install update rebuilds through the command (a DMG copy
# keeps the installer's update path): src/clean_voice/updates.py MARKER_NAME. It holds the SHA-256 of the app's main
# binary and lives in Application Support: a file added inside the signed bundle would break its signature, and a DMG
# copied over this one has another binary, so the hash no longer matches.
write_marker() {
    local dir="$HOME/Library/Application Support/Clean Voice" sum
    sum="$(shasum -a 256 "$1/Contents/MacOS/Clean Voice" | cut -d ' ' -f 1)"
    if ! { mkdir -p "$dir" && printf '%s\n' "$sum" >"$dir/installed-by-command"; }; then
        log "Could not write the installed-by-command marker"
    fi
}

# ---- The install ------------------------------------------------------------------------------------------------

cleanup() {
    cd "$HOME" 2>/dev/null
    rm -rf "$WORK"
}

main() {
    init_vars
    trap cleanup EXIT
    mkdir -p "$(dirname "$LOG")"
    printf 'Clean Voice install %s %s\n' "$VERSION" "$(date)" >"$LOG"

    step 1 'Checking your Mac'
    check_macos "$(sw_vers -productVersion)"
    if [ -z "$UPDATE" ] && app_running; then
        die 'Clean Voice is open. Close it, then run the command again.'
    fi

    step 2 'Getting the build tools'
    rm -rf "$WORK"
    mkdir -p "$WORK" || die_step 'could not make the setup folder'
    ensure_tools
    local archive
    archive="$(uv_archive)" || die 'Clean Voice needs a Mac with an Apple chip or an Intel chip.'
    fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$archive.tar.gz" "$WORK/uv.tar.gz"
    check_sha256 "$WORK/uv.tar.gz" "$(uv_sha256 "$archive")"
    mkdir -p "$WORK/uv"
    run_logged tar -xzf "$WORK/uv.tar.gz" -C "$WORK/uv" --strip-components 1 || die_step 'could not unpack uv'

    step 3 'Getting Clean Voice'
    fetch "$BASE/Clean-Voice-source.zip" "$WORK/source.zip"
    check_sha256 "$WORK/source.zip" "$SOURCE_SHA256"
    run_logged unzip -q "$WORK/source.zip" -d "$WORK/source" || die_step 'could not unpack the source'

    step 4 'Getting Python'
    # Source: docs.astral.sh/uv/concepts/python-versions. 3.12 because Briefcase builds for the running Python
    # (briefcase/platforms/base.py:727-735); --seed because Briefcase's own venv needs pip
    # (briefcase/integrations/virtual_environment/venv.py:48-72).
    export UV_PYTHON_PREFERENCE=only-managed
    export UV_PYTHON_INSTALL_DIR="$WORK/python"
    export UV_CACHE_DIR="$WORK/uvcache"
    export BRIEFCASE_HOME="$WORK/briefcase"
    local uv="$WORK/uv/uv" python="$WORK/venv/bin/python"
    run_logged "$uv" python install 3.12 || die_step 'could not get Python'
    run_logged "$uv" venv --python 3.12 --seed "$WORK/venv" || die_step 'could not make the Python environment'
    run_logged "$uv" pip install --python "$python" "briefcase==$BRIEFCASE_PIN" || die_step 'could not get Briefcase'

    step 5 'Building Clean Voice (a few minutes)'
    # Source: briefcase/platforms/macOS/app.py:152-162 (build ad-hoc signs the bundle); universal_build in pyproject.toml.
    cd "$WORK/source" || die_step 'could not open the source'
    run_logged "$python" -m briefcase create macOS --no-input || die_step 'building (create)'
    run_logged "$python" -m briefcase build macOS --no-input || die_step 'building (build)'
    local built="$WORK/source/build/clean-voice/macos/app/$APP_NAME"
    [ -d "$built" ] || die_step 'the built app is missing'

    step 6 'Installing'
    local target
    target="$(pick_target)"
    swap_app "$built" "$target"
    write_marker "$target"

    step 7 'Finishing'
    if [ -n "$UPDATE" ]; then
        open "$target" </dev/null >/dev/null 2>&1
        say 'Clean Voice was updated and is opening again.'
    else
        say "Clean Voice is installed in $(dirname "$target"). Open it from there."
    fi
}

if [ -z "${CLEAN_VOICE_TEST_IMPORT:-}" ]; then main "$@"; fi
