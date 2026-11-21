"""The Windows install scripts ``tools/install/install.ps1`` and ``uninstall.ps1`` (B019, step 2).

They run under Windows PowerShell 5.1 (``powershell.exe``, never ``pwsh``), dot-sourced with CLEAN_VOICE_TEST_IMPORT set
so they only define their functions. Each test writes a small scenario script, runs it and reads what it printed: the
seams the scripts expose (``Start-Process``, ``Move-CvTree``, ``Start-CvApp``, ``Get-CvFile``, ...) are replaced by
fakes, and every path, registry key and the Start menu are scratch places. Nothing here times anything: a test
asserts on exit code, output and the state left behind (docs/checklists/tests-timing.md).
"""

import os
import re
import subprocess
import sys
import tomllib
import uuid
from pathlib import Path

import pytest
import shell_env

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "tools" / "install" / "install.ps1"
UNINSTALL = ROOT / "tools" / "install" / "uninstall.ps1"
POWERSHELL = shell_env.powershell_path() or Path("powershell.exe")

DECLINED = "An older Clean Voice couldn't be removed"
REOPENED = "Your old version is back and was reopened."
NOTHING_CHANGED = "Your old version is back. Nothing was changed."

PRELUDE = r"""
$ErrorActionPreference = 'Stop'
$env:CLEAN_VOICE_TEST_IMPORT = '1'
. '__INSTALL__'
$t = '__TMP__'
$reg = '__REG__'
New-Item -Path $reg -Force | Out-Null

function New-TestCx {
    param([hashtable]$Extra = @{})
    $o = @{
        Work = "$t\work"; Folder = "$t\apps\Clean Voice"; Log = "$t\log.txt"
        UninstallRoot = "$reg\Uninstall"; RunKey = "$reg\Run"; StartMenu = "$t\start"
        OldMsiRoots = @(@{ Path = "$reg\UserMsi"; Scope = 'User' }, @{ Path = "$reg\MachineMsi"; Scope = 'Machine' })
        SacPath = "$reg\CI"; StatusFile = "$t\status.txt"; InstallerData = "$reg\InstallerData"
    }
    foreach ($k in $Extra.Keys) { $o[$k] = $Extra[$k] }
    $global:TestCx = New-CvContext $o
    $global:TestCx
}
function Set-File { param($Path, $Text) New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null; [IO.File]::WriteAllText($Path, $Text) }
function Get-Text { param($Path) if (Test-Path -LiteralPath $Path) { [IO.File]::ReadAllText($Path) } else { 'none' } }
function Set-OldApp { param($Cx) Set-File (Join-Path $Cx.Folder 'Clean Voice.exe') 'old' }
function Set-NewApp {
    param($Cx)
    Set-File (Join-Path $Cx.NewTree 'Clean Voice.exe') 'new'
    Set-File (Join-Path $Cx.Source 'tools\install\uninstall.ps1') ([IO.File]::ReadAllText('__UNINSTALL__'))
}
function Add-Msi {
    param([string]$Scope, [string]$Code, [string]$Name = 'Clean Voice', [int]$Installer = 1)
    $key = "$reg\${Scope}Msi\$Code"
    New-Item -Path $key -Force | Out-Null
    New-ItemProperty -Path $key -Name DisplayName -Value $Name -Force | Out-Null
    New-ItemProperty -Path $key -Name WindowsInstaller -Value $Installer -PropertyType DWord -Force | Out-Null
}
function Add-Owner {
    param([string]$Sid, [string]$Code)
    New-Item -Path "$reg\InstallerData\$Sid\Products\$(Get-CvPackedGuid $Code)" -Force | Out-Null
}

# Fakes. Start-Process records what it was asked for and what the folders looked like at that moment.
$global:Calls = New-Object System.Collections.ArrayList
$global:Exit = 0
$global:Cancel = $false
function Start-Process {
    param([string]$FilePath, $ArgumentList, [string]$Verb, [string]$WorkingDirectory, [switch]$Wait, [switch]$PassThru)
    $c = $global:TestCx
    [void]$global:Calls.Add(@{
        File = $FilePath; Args = "$ArgumentList"; Verb = $Verb; Wait = [bool]$Wait
        Folder = (Get-Text (Join-Path $c.Folder 'Clean Voice.exe')); Old = (Get-Text (Join-Path $c.Old 'Clean Voice.exe'))
    })
    if ($FilePath -ne 'msiexec') { return }
    # A real per-user MSI removal can leave an empty folder at its install path (CI run 44898585914).
    if ($global:MsiLeavesFolder -and -not $Verb) { New-Item -ItemType Directory -Path $c.Folder -Force | Out-Null }
    if ($global:Cancel) { throw (New-Object System.ComponentModel.Win32Exception 1223) }
    # A removal that wipes the new files, as the real one did in CI run 44902719464.
    if ($global:WipeFolderOn -eq "$Verb") { Remove-Item -LiteralPath $c.Folder -Recurse -Force }
    [pscustomobject]@{ ExitCode = $global:Exit }
}
$global:Launched = New-Object System.Collections.ArrayList
function Start-CvApp { param([string]$Exe) [void]$global:Launched.Add($Exe) }
function Wait-CvEnter { }

$global:FailMove = 0
function Move-CvTree {
    param([string]$From, [string]$To)
    $c = $global:TestCx
    if ($global:FailMove -eq 1 -and $From -eq $c.Folder) { throw 'forced move failure (1)' }
    if ($global:FailMove -eq 3 -and $From -eq $c.NewTree) { throw 'forced move failure (3)' }
    Move-Item -LiteralPath $From -Destination $To
}

function Show-Calls {
    foreach ($c in $global:Calls) {
        if ($c.File -eq 'msiexec') { "CALL|$($c.Args)|verb=$($c.Verb)|wait=$($c.Wait)|folder=$($c.Folder)|old=$($c.Old)" }
    }
}
function Try-Swap {
    param($Cx, $Msi)
    try { $r = Invoke-CvSwap $Cx $Msi; "SWAP|ok|machineRemoved=$r" } catch { "SWAP|error|$($_.Exception.Message)" }
}
function Show-State {
    param($Cx)
    "STATE|folder=$(Get-Text (Join-Path $Cx.Folder 'Clean Voice.exe'))|old=$(Test-Path -LiteralPath $Cx.Old)|newtree=$(Test-Path -LiteralPath $Cx.NewTree)" +
        "|key=$(Test-Path -LiteralPath (Join-Path $Cx.UninstallRoot $Cx.UninstallName))|status=$(Get-Text $Cx.StatusFile)|launched=$($global:Launched.Count)"
}
"""

TEMPLATE = """{prelude}
try {{
{body}
}} finally {{
    Remove-Item -LiteralPath $reg -Recurse -Force -ErrorAction SilentlyContinue
}}
"""


@pytest.fixture
def ps(tmp_path: Path):
    """Run a scenario in Windows PowerShell 5.1 and return what it printed (stdout lines)."""
    # Inside a Claude session the safety net refuses to start PowerShell: these run in CI (tests/shell_env.py).
    if sys.platform != "win32":
        pytest.skip("Windows PowerShell scripts: Windows only")
    shell_env.require("Windows PowerShell 5.1 is needed", shell_env.powershell_path() is not None)
    reg = rf"HKCU:\Software\CleanVoiceTest\{uuid.uuid4().hex}"

    def run(body: str, *, env: dict[str, str] | None = None, timeout: float = 180) -> list[str]:
        prelude = (
            PRELUDE.replace("__INSTALL__", str(INSTALL))
            .replace("__UNINSTALL__", str(UNINSTALL))
            .replace("__TMP__", str(tmp_path))
            .replace("__REG__", reg)
        )
        script = tmp_path / "scenario.ps1"
        script.write_text(TEMPLATE.format(prelude=prelude, body=body), encoding="utf-8-sig")
        full_env = {k: v for k, v in os.environ.items() if not k.startswith("CLEAN_VOICE_")}
        full_env.update(env or {})
        proc = subprocess.run(
            [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env,
            check=False,
        )
        assert proc.returncode == 0, f"scenario failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        return proc.stdout.splitlines()

    yield run
    subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-Command",
            f"Remove-Item -LiteralPath '{reg}' -Recurse -Force -ErrorAction SilentlyContinue",
        ],
        capture_output=True,
        check=False,
    )


def lines_with(output: list[str], prefix: str) -> list[str]:
    return [line for line in output if line.startswith(prefix)]


def state(output: list[str]) -> dict[str, str]:
    (line,) = lines_with(output, "STATE|")
    return dict(part.split("=", 1) for part in line.split("|")[1:])


# ---- The files themselves ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("script", [INSTALL, UNINSTALL], ids=["install", "uninstall"])
def test_ascii_only_no_bom(script: Path) -> None:
    # B019-2: a guest's PowerShell 5.1 reads a script from the web as ANSI, so any non-ASCII byte becomes garbage.
    data = script.read_bytes()
    assert all(b < 128 for b in data), [i for i, b in enumerate(data) if b >= 128][:5]
    assert not data.startswith(b"\xef\xbb\xbf")


def test_both_parse_clean_and_install_has_no_exit(ps) -> None:
    out = ps(
        r"""
foreach ($f in '__INSTALL__', '__UNINSTALL__') {
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($f, [ref]$tokens, [ref]$errors)
    $exits = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.ExitStatementAst] }, $true)
    "PARSE|$(Split-Path -Leaf $f)|errors=$($errors.Count)|exits=$($exits.Count)"
}
""".replace("__INSTALL__", str(INSTALL)).replace("__UNINSTALL__", str(UNINSTALL))
    )
    assert lines_with(out, "PARSE|") == ["PARSE|install.ps1|errors=0|exits=0", "PARSE|uninstall.ps1|errors=0|exits=0"]


def test_main_is_called_on_the_last_line() -> None:
    last = INSTALL.read_text(encoding="ascii").rstrip().splitlines()[-1]
    assert last == "if (-not $env:CLEAN_VOICE_TEST_IMPORT) { Invoke-CvInstall }"


def test_placeholders_appear_once() -> None:
    text = INSTALL.read_text(encoding="ascii")
    assert text.count("@@VERSION@@") == 1
    assert text.count("@@SOURCE_SHA256@@") == 1


def test_briefcase_home_is_made_before_use() -> None:
    # CI run 44893694468: Briefcase refuses a BRIEFCASE_HOME that does not exist (briefcase/commands/base.py:211-217).
    text = INSTALL.read_text(encoding="ascii")
    assert re.search(r"\$env:BRIEFCASE_HOME = \(New-Item -ItemType Directory -Force", text)


def test_pins_equal_pyproject(ps) -> None:
    dev = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["dependency-groups"]["dev"]
    (pin,) = [d.split("==")[1] for d in dev if d.startswith("briefcase==")]
    out = ps("'PIN|' + (Get-CvPins).Briefcase")
    assert out == [f"PIN|{pin}"]
    text = INSTALL.read_text(encoding="ascii")
    assert re.search(r"UvSha256\s*=\s*'[0-9a-f]{64}'", text)
    assert re.search(r"MinGitSha256\s*=\s*'[0-9a-f]{64}'", text)


# ---- Downloads and checks ---------------------------------------------------------------------------------------


def test_checksum_good_and_bad(ps) -> None:
    out = ps(
        r"""
Set-File "$t\f.bin" 'hello'
$good = (Get-FileHash -Algorithm SHA256 -LiteralPath "$t\f.bin").Hash.ToLower()
try { Assert-CvHash "$t\f.bin" $good; 'GOOD|ok' } catch { "GOOD|$($_.Exception.Message)" }
try { Assert-CvHash "$t\f.bin" ('0' * 64); 'BAD|ok' } catch { "BAD|$($_.Exception.Message)" }
"""
    )
    assert out == ["GOOD|ok", "BAD|CVPLAIN:The download didn't match what was published, so nothing was installed."]


def test_base_url_copy_and_the_missing_file_line(ps, tmp_path: Path) -> None:
    served = tmp_path / "served"
    served.mkdir()
    (served / "Clean-Voice-source.zip").write_bytes(b"zip")
    out = ps(
        r"""
function Invoke-WebRequest {
    param($Uri, $OutFile, [switch]$UseBasicParsing)
    $name = $Uri.Substring($Uri.LastIndexOf('/') + 1)
    $from = Join-Path "$t\served" $name
    if (-not (Test-Path -LiteralPath $from)) { throw '404' }
    Copy-Item -LiteralPath $from -Destination $OutFile
}
$cx = New-TestCx
"BASE|$($cx.Base)"
Get-CvFile $cx ($cx.Base + '/Clean-Voice-source.zip') "$t\got.zip"
"GOT|$(Get-Text "$t\got.zip")"
try { Get-CvFile $cx ($cx.Base + '/nothing.zip') "$t\nothing.zip"; 'MISSING|ok' } catch { "MISSING|$($_.Exception.Message)" }
""",
        env={"CLEAN_VOICE_BASE": "http://127.0.0.1:9/"},
    )
    assert out == [
        "BASE|http://127.0.0.1:9",
        "GOT|zip",
        "MISSING|CVPLAIN:Couldn't reach GitHub. Check your internet connection and run the command again.",
    ]


def test_default_base_is_githubs_latest_download(ps) -> None:
    out = ps("'BASE|' + (New-TestCx).Base", env={"CLEAN_VOICE_BASE": ""})
    assert out == ["BASE|https://github.com/paulcaesarflores/clean-voice/releases/latest/download"]


def test_unsupported_windows_lines(ps) -> None:
    out = ps(
        r"""
"ARM|$(Get-CvPlatformProblem 'ARM64' 22631)"
"OLD|$(Get-CvPlatformProblem 'AMD64' 9600)"
"OK|$(Get-CvPlatformProblem 'AMD64' 22631)|"
"""
    )
    line = "Clean Voice needs 64-bit Windows 10 or 11 (Intel/AMD)."
    assert out == [f"ARM|{line}", f"OLD|{line}", "OK||"]


def test_smart_app_control_warns_and_continues_only_when_on(ps) -> None:
    out = ps(
        r"""
$cx = New-TestCx
"NONE|$(Get-CvSacState $cx)|"
New-Item -Path $cx.SacPath -Force | Out-Null
New-ItemProperty -Path $cx.SacPath -Name VerifiedAndReputablePolicyState -Value 1 -PropertyType DWord -Force | Out-Null
"STATE|$(Get-CvSacState $cx)"
'ON-BEGIN'
Show-CvSacWarning $cx (Get-CvSacState $cx)
'ON-END'
Set-ItemProperty -Path $cx.SacPath -Name VerifiedAndReputablePolicyState -Value 0
'OFF-BEGIN'
Show-CvSacWarning $cx (Get-CvSacState $cx)
Show-CvSacWarning $cx $null
'OFF-END'
"""
    )
    on = out[out.index("ON-BEGIN") + 1 : out.index("ON-END")]
    assert out[:2] == ["NONE||", "STATE|1"]
    assert len(on) == 1
    assert on[0].startswith("Windows' Smart App Control is on and may block Clean Voice, because we can't sign it.")
    assert "Smart App Control settings > Off" in on[0] and on[0].endswith("Installing anyway.")
    assert out[out.index("OFF-BEGIN") + 1 : out.index("OFF-END")] == []


# ---- The whole run, with the downloads and Briefcase faked --------------------------------------------------------

FLOW = r"""
$global:Urls = New-Object System.Collections.ArrayList
$global:Natives = New-Object System.Collections.ArrayList
function Get-CvFile { param($Cx, $Url, $Dest) [void]$global:Urls.Add($Url); Set-File $Dest 'x' }
function Assert-CvHash { param($Path, $Expected) }
function Expand-CvZip {
    param($Zip, $Dest)
    New-Item -ItemType Directory -Path $Dest -Force | Out-Null
    if ($Dest -eq $global:TestCx.Source) { Set-NewApp $global:TestCx }
}
function Invoke-CvNative { param($Cx, $File, $Arguments) [void]$global:Natives.Add(($Arguments -join ' ')) }
function Get-CvProcess { $global:Running }
$global:Running = $null
"""


def test_a_clean_install_runs_every_step_and_registers(ps) -> None:
    out = ps(
        FLOW
        + r"""
$cx = New-TestCx
Invoke-CvInstall $cx
Show-State $cx
"LNK|$(Test-Path -LiteralPath "$t\start\Clean Voice\Clean Voice.lnk")|WORK|$(Test-Path -LiteralPath $cx.Work)"
"MARKER|" + (Get-Text (Join-Path $cx.Folder 'installed-by-command'))
"NATIVES|" + ($global:Natives -join ';')
"URLS|" + (($global:Urls | ForEach-Object { $_.Substring($_.LastIndexOf('/') + 1) }) -join ';')
"""
    )
    assert [line for line in out if line.startswith("Step ")] == [
        "Step 1 of 8: Checking your PC",
        "Step 2 of 8: Getting the build tools",
        "Step 3 of 8: Getting Clean Voice",
        "Step 4 of 8: Getting Python",
        "Step 5 of 8: Building Clean Voice (a few minutes)",
        "Step 6 of 8: Looking for an older copy",
        "Step 7 of 8: Installing",
        "Step 8 of 8: Finishing",
    ]
    assert "Clean Voice is installed. Open it from the Start menu." in out
    s = state(out)
    assert (s["folder"], s["old"], s["key"], s["launched"]) == ("new", "False", "True", "0")
    marker = next(line for line in out if line.startswith("MARKER|"))
    assert marker != "MARKER|none"  # the app finds this file to update through the command
    assert "LNK|True|WORK|False" in out  # the setup folder is removed at the end
    (natives,) = lines_with(out, "NATIVES|")
    assert natives.split("|")[1].split(";")[0] == "python install 3.12"
    assert "briefcase create windows --no-input" in natives and "briefcase build windows --no-input" in natives
    (urls,) = lines_with(out, "URLS|")
    assert urls.endswith("Clean-Voice-source.zip")


def test_running_clean_voice_stops_a_normal_install(ps) -> None:
    out = ps(
        FLOW
        + r"""
$cx = New-TestCx
Set-OldApp $cx
$global:Running = 'something'
Invoke-CvInstall $cx
Show-State $cx
"""
    )
    assert "Clean Voice is open. Close it, then run the command again." in out
    assert state(out)["folder"] == "old"
    assert state(out)["status"] == "failed"


def test_smart_app_control_on_still_installs(ps) -> None:
    out = ps(
        FLOW
        + r"""
$cx = New-TestCx
New-Item -Path $cx.SacPath -Force | Out-Null
New-ItemProperty -Path $cx.SacPath -Name VerifiedAndReputablePolicyState -Value 2 -PropertyType DWord -Force | Out-Null
Invoke-CvInstall $cx
Show-State $cx
"""
    )
    assert any(line.startswith("Windows' Smart App Control is on") for line in out)
    assert state(out)["folder"] == "new"


def test_any_thrown_error_writes_failed_and_names_the_step(ps) -> None:
    out = ps(
        FLOW
        + r"""
function Assert-CvHash { param($Path, $Expected) throw 'boom' }
$cx = New-TestCx
Set-OldApp $cx
Invoke-CvInstall $cx
Show-State $cx
"WORK|$(Test-Path -LiteralPath $cx.Work)"
"""
    )
    assert any(
        line.startswith("Something went wrong at step 2 (Getting the build tools). Nothing was changed. The log is at ")
        for line in out
    )
    s = state(out)
    assert (s["folder"], s["status"]) == ("old", "failed")
    assert "WORK|False" in out


def test_a_checksum_mismatch_stops_before_unzipping(ps) -> None:
    out = ps(
        FLOW
        + r"""
function Assert-CvHash { param($Path, $Expected) if ($Path -like '*source.zip') { Stop-Install "The download didn't match what was published, so nothing was installed." } }
$global:Unzipped = 0
function Expand-CvZip { param($Zip, $Dest) $global:Unzipped++ }
$cx = New-TestCx
Invoke-CvInstall $cx
"UNZIPPED|$global:Unzipped"
Show-State $cx
"""
    )
    assert "The download didn't match what was published, so nothing was installed." in out
    assert "UNZIPPED|2" in out  # uv and MinGit only, never the source
    assert state(out)["folder"] == "none"


# ---- Update mode ---------------------------------------------------------------------------------------------------


def test_update_mode_writes_swap_waits_for_the_pid_skips_the_msi_step_and_starts_the_new_app(ps) -> None:
    out = ps(
        FLOW
        + r"""
$global:FindCalls = 0
function Find-CvOldMsi { param($Cx) $global:FindCalls++; @{ User = @(); Machine = @() } }
$p = Microsoft.PowerShell.Management\Start-Process -FilePath cmd.exe -ArgumentList '/c', 'exit' -PassThru -WindowStyle Hidden
$p.WaitForExit()
$cx = New-TestCx @{ Update = $true; WaitPid = $p.Id }
Set-OldApp $cx
Invoke-CvInstall $cx
Show-State $cx
"FIND|$global:FindCalls"
"LAUNCHED|$($global:Launched -join ';')"
"""
    )
    s = state(out)
    assert (s["status"], s["folder"], s["launched"]) == ("swap", "new", "1")
    assert "FIND|0" in out
    assert not any(line.startswith("Step 6 of 8") for line in out)
    assert any(line.startswith("LAUNCHED|") and line.endswith(r"Clean Voice\Clean Voice.exe") for line in out)
    assert "Clean Voice was updated and is opening again." in out


def test_update_mode_says_so_when_the_app_does_not_close(ps) -> None:
    out = ps(
        FLOW
        + r"""
$cx = New-TestCx @{ Update = $true; WaitPid = $PID; WaitSeconds = 1 }
Set-OldApp $cx
Invoke-CvInstall $cx
Show-State $cx
"""
    )
    assert "Clean Voice didn't close, so the old version is still in place." in out
    s = state(out)
    # "swap" was written first (the app reads it), then the failure replaced it; nothing was touched.
    assert (s["folder"], s["old"], s["status"], s["launched"]) == ("old", "False", "failed", "0")


# ---- The old MSI ----------------------------------------------------------------------------------------------------

GUID_U = "{11111111-1111-1111-1111-111111111111}"
GUID_M = "{22222222-2222-2222-2222-222222222222}"


def swap_scenario(extra: str = "", setup: str = "", cx: str = "New-TestCx") -> str:
    return (
        rf"""
$cx = {cx}
Set-OldApp $cx
Set-NewApp $cx
{setup}
Try-Swap $cx (Find-CvOldMsi $cx)
Show-Calls
Show-State $cx
"""
        + extra
    )


def test_no_old_msi_calls_nothing(ps) -> None:
    out = ps(
        swap_scenario(
            setup="Add-Msi 'User' '{AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA}' 'Other App'; Add-Msi 'User' 'CleanVoice' 'Clean Voice'; Add-Msi 'Machine' '"
            + GUID_M
            + "' 'Clean Voice' 0"
        )
    )
    assert lines_with(out, "CALL|") == []
    assert "SWAP|ok|machineRemoved=False" in out
    assert state(out)["folder"] == "new"


def test_a_per_user_msi_is_removed_directly_and_a_per_machine_one_through_the_admin_prompt(ps) -> None:
    out = ps(swap_scenario(setup=f"Add-Msi 'User' '{GUID_U}'; Add-Msi 'Machine' '{GUID_M}'; $global:Exit = 3010"))
    calls = lines_with(out, "CALL|")
    assert len(calls) == 2
    assert calls[0].startswith(f"CALL|/x {GUID_U} /qn /l*v ")
    assert "|verb=|wait=True|" in calls[0]
    assert calls[1].startswith(f"CALL|/x {GUID_M} /qn|verb=RunAs|wait=True|")  # exit 3010 (restart needed) succeeds
    assert "SWAP|ok|machineRemoved=True" in out


def test_msi_removal_order_keeps_a_working_clean_voice(ps) -> None:
    out = ps(swap_scenario(setup=f"Add-Msi 'User' '{GUID_U}'; Add-Msi 'Machine' '{GUID_M}'"))
    user_call, machine_call = lines_with(out, "CALL|")
    # Per user: removed while the old folder is renamed aside (its files are in the folder the new app goes to).
    assert user_call.endswith("|folder=none|old=old")
    # Per machine: only after the new app is in place.
    assert machine_call.endswith("|folder=new|old=old")
    s = state(out)
    assert (s["folder"], s["old"]) == ("new", "False")


@pytest.mark.parametrize("failure", ["decline", "exit1"])
def test_a_declined_per_machine_prompt_deletes_the_new_tree_and_restores_the_old_copy(ps, failure: str) -> None:
    setup = f"Add-Msi 'Machine' '{GUID_M}'; " + (
        "$global:Cancel = $true" if failure == "decline" else "$global:Exit = 1"
    )
    out = ps(swap_scenario(setup=setup))
    assert any(line.startswith("SWAP|error|CVPLAIN:" + DECLINED) for line in out)
    assert "Remove 'Clean Voice' in Settings > Installed apps, then run the command again." in "".join(out)
    s = state(out)
    assert (s["folder"], s["old"], s["key"]) == ("old", "False", "True")  # old copy back, registered again


@pytest.mark.parametrize("failure", ["decline", "exit1"])
def test_a_failed_per_user_removal_stops_with_the_old_copy_intact(ps, failure: str) -> None:
    setup = f"Add-Msi 'User' '{GUID_U}'; " + (
        "$global:Cancel = $true" if failure == "decline" else "$global:Exit = 1603"
    )
    out = ps(swap_scenario(setup=setup))
    assert any(line.startswith("SWAP|error|CVPLAIN:" + DECLINED) for line in out)
    s = state(out)
    assert (s["folder"], s["old"], s["key"], s["launched"]) == ("old", "False", "True", "0")


USER_SID = "S-1-5-21-1-2-3-1001"


def test_the_packed_product_code_matches_windows_installers_own_form(ps) -> None:
    out = ps(r"""Get-CvPackedGuid '{77661479-C123-4B94-A83F-81B3AC874517}'""")
    assert out == ["97416677321C49B48AF3183BCA785471"]


def test_a_per_user_product_registered_under_hklm_is_removed_before_the_move(ps) -> None:
    # CI run 44902719464: v0.2.0 installed per user by an administrator had its Uninstall key in HKLM but its files in
    # the per-user folder; classed by hive it was removed after the move and deleted the new app.
    out = ps(
        swap_scenario(
            setup=f"Add-Msi 'Machine' '{GUID_M}'; Add-Owner '{USER_SID}' '{GUID_M}'; $global:MsiLeavesFolder = $true"
        )
        + "\n(Find-CvOldMsi $cx).User -join ','\n(Find-CvOldMsi $cx).Machine.Count"
    )
    (call,) = lines_with(out, "CALL|")
    assert call.startswith(f"CALL|/x {GUID_M} /qn /l*v ") and "|verb=|" in call
    assert call.endswith("|folder=none|old=old")
    assert "SWAP|ok|machineRemoved=False" in out
    assert state(out)["folder"] == "new"
    assert out[-2:] == [GUID_M, "0"]


def test_a_product_owned_by_the_system_account_stays_per_machine(ps) -> None:
    out = ps(swap_scenario(setup=f"Add-Msi 'Machine' '{GUID_M}'; Add-Owner 'S-1-5-18' '{GUID_M}'"))
    (call,) = lines_with(out, "CALL|")
    assert call.startswith(f"CALL|/x {GUID_M} /qn|verb=RunAs|") and call.endswith("|folder=new|old=old")
    assert "SWAP|ok|machineRemoved=True" in out


def test_the_last_exe_check_runs_after_every_removal(ps) -> None:
    # The per-machine removal is the last one; when it deletes the new files the swap rolls back instead of "installed".
    out = ps(swap_scenario(setup=f"Add-Msi 'Machine' '{GUID_M}'; $global:WipeFolderOn = 'RunAs'"))
    assert any(line.startswith("SWAP|error|CVPLAIN:") and NOTHING_CHANGED in line for line in out)
    s = state(out)
    assert (s["folder"], s["old"], s["key"]) == ("old", "False", "True")  # the old copy is back and registered


def test_every_msiexec_call_waits_and_reads_the_exit_code() -> None:
    # msiexec is a GUI-subsystem program: called bare, PowerShell would not wait and the removal would run beside the swap.
    text = INSTALL.read_text(encoding="ascii")
    calls = re.findall(r"^[^#\n]*msiexec[^\n]*$", text, flags=re.MULTILINE)
    assert calls
    for call in calls:
        assert "Start-Process" in call and "-Wait" in call and "-PassThru" in call, call


def test_a_folder_left_by_the_msi_removal_does_not_swallow_the_new_tree(ps) -> None:
    # Move-Item into an existing folder nests the tree ("...\Clean Voice\src\Clean Voice.exe"): the guest would be left
    # with no app and a "Clean Voice is installed" line.
    out = ps(swap_scenario(setup=f"Add-Msi 'User' '{GUID_U}'; $global:MsiLeavesFolder = $true"))
    assert "SWAP|ok|machineRemoved=False" in out
    s = state(out)
    assert (s["folder"], s["old"], s["newtree"]) == ("new", "False", "False")


def test_a_move_that_leaves_no_exe_is_rolled_back(ps) -> None:
    out = ps(
        swap_scenario(
            setup=(
                "function Move-CvTree { param($From, $To) if ($From -eq $global:TestCx.NewTree) "
                "{ New-Item -ItemType Directory -Path $To | Out-Null } else { Move-Item -LiteralPath $From -Destination $To } }"
            )
        )
    )
    assert any(line.startswith("SWAP|error|") for line in out)
    s = state(out)
    assert (s["folder"], s["old"], s["key"]) == ("old", "False", "True")


def test_the_swap_replaces_the_folder_and_deletes_a_stale_old(ps) -> None:
    out = ps(swap_scenario(setup="Set-File (Join-Path $cx.Old 'stale.txt') 'stale'"))
    assert "SWAP|ok|machineRemoved=False" in out
    s = state(out)
    assert (s["folder"], s["old"], s["newtree"]) == ("new", "False", "False")


def test_the_run_value_is_rewritten_only_after_a_per_machine_removal_and_only_if_present(ps) -> None:
    out = ps(
        r"""
$cx = New-TestCx
Set-NewApp $cx
New-Item -Path $cx.RunKey -Force | Out-Null
Set-File (Join-Path $cx.Folder 'Clean Voice.exe') 'new'
function Get-Run { (Get-ItemProperty -Path $cx.RunKey -Name 'Clean Voice' -ErrorAction SilentlyContinue).'Clean Voice' }
"ABSENT|rewrite|" + $(Register-CvInstall $cx '1.0.0' $true; Get-Run)
New-ItemProperty -Path $cx.RunKey -Name 'Clean Voice' -Value '"C:\Program Files\Clean Voice\Clean Voice.exe" --minimized' -Force | Out-Null
Register-CvInstall $cx '1.0.0' $false
"KEPT|" + (Get-Run)
Register-CvInstall $cx '1.0.0' $true
"REWRITTEN|" + (Get-Run)
"EXE|" + (Join-Path $cx.Folder 'Clean Voice.exe')
"""
    )
    exe = next(line for line in out if line.startswith("EXE|")).split("|", 1)[1]
    assert "ABSENT|rewrite|" in out
    assert r'KEPT|"C:\Program Files\Clean Voice\Clean Voice.exe" --minimized' in out
    assert f'REWRITTEN|"{exe}" --minimized' in out


# ---- Rollback ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fail_at", [1, 3])
@pytest.mark.parametrize("update", [False, True], ids=["normal", "update"])
def test_a_move_failure_restores_the_old_folder_and_reregisters_it(ps, fail_at: int, update: bool) -> None:
    cx = "New-TestCx @{ Update = $true }" if update else "New-TestCx"
    out = ps(swap_scenario(setup=f"$global:FailMove = {fail_at}", cx=cx))
    (error,) = lines_with(out, "SWAP|error|")
    s = state(out)
    assert (s["folder"], s["old"], s["key"]) == ("old", "False", "True")
    if update:
        assert error.endswith(REOPENED)
        assert (s["launched"], s["status"]) == ("1", "failed")  # the old exe was started once, from the restored folder
        assert any(line.startswith("STATE|") for line in out)
    else:
        assert error.endswith(NOTHING_CHANGED)
        assert (s["launched"], s["status"]) == ("0", "none")


def test_the_relaunch_names_the_restored_exe(ps) -> None:
    out = ps(
        swap_scenario(setup="$global:FailMove = 3", cx="New-TestCx @{ Update = $true }")
        + "\n'LAUNCHED|' + ($global:Launched -join ';') + '|FOLDER|' + $cx.Folder"
    )
    (line,) = lines_with(out, "LAUNCHED|")
    launched, folder = line.split("|")[1], line.split("|")[3]
    assert launched == folder + r"\Clean Voice.exe"


# ---- Registration --------------------------------------------------------------------------------------------------------


def test_registration_values_start_menu_and_the_uninstall_script_the_string_names(ps) -> None:
    out = ps(
        r"""
$cx = New-TestCx
Set-NewApp $cx
Set-File (Join-Path $cx.Folder 'Clean Voice.exe') 'new'
Register-CvInstall $cx '9.8.7'
$p = Get-ItemProperty -Path (Join-Path $cx.UninstallRoot $cx.UninstallName)
foreach ($n in 'DisplayName', 'DisplayVersion', 'Publisher', 'DisplayIcon', 'InstallLocation', 'UninstallString', 'NoModify', 'NoRepair') { "VAL|$n|$($p.$n)" }
$shell = New-Object -ComObject WScript.Shell
"LINK|" + $shell.CreateShortcut("$t\start\Clean Voice\Clean Voice.lnk").TargetPath
"FOLDER|" + $cx.Folder
"""
    )
    values = dict(line.split("|", 2)[1:] for line in out if line.startswith("VAL|"))
    folder = next(line for line in out if line.startswith("FOLDER|")).split("|", 1)[1]
    script = folder + r"\Uninstall Clean Voice.ps1"
    assert values["DisplayName"] == "Clean Voice" and values["Publisher"] == "Clean Voice"
    assert values["DisplayVersion"] == "9.8.7"
    assert values["DisplayIcon"] == values["InstallLocation"] + r"\Clean Voice.exe" == folder + r"\Clean Voice.exe"
    assert values["NoModify"] == "1" and values["NoRepair"] == "1"
    assert values["UninstallString"] == f'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{script}"'
    assert Path(script).read_text(encoding="ascii") == UNINSTALL.read_text(encoding="ascii")
    assert f"LINK|{folder}\\Clean Voice.exe" in out


# ---- Uninstall -----------------------------------------------------------------------------------------------------------

UNINSTALL_PRELUDE = r"""
$env:CLEAN_VOICE_TEST_IMPORT = '1'
. '__UNINSTALL__'
function Get-CvProcess { $global:Running }
$global:Running = $null
function New-UCx {
    New-CvUninstallContext @{ Folder = "$t\apps\Clean Voice"; UninstallRoot = "$reg\Uninstall"; RunKey = "$reg\Run"; StartMenu = "$t\start" }
}
function Install-Fake {
    param($Cx, [string]$RunValue)
    Set-File (Join-Path $Cx.Folder 'Clean Voice.exe') 'x'
    New-Item -Path (Join-Path $Cx.UninstallRoot $Cx.UninstallName) -Force | Out-Null
    Set-File (Join-Path $Cx.StartMenu 'Clean Voice\Clean Voice.lnk') 'lnk'
    New-Item -Path $Cx.RunKey -Force | Out-Null
    if ($RunValue) { New-ItemProperty -Path $Cx.RunKey -Name 'Clean Voice' -Value $RunValue -Force | Out-Null }
}
function Show-U {
    param($Cx)
    "U|folder=$(Test-Path -LiteralPath $Cx.Folder)|key=$(Test-Path -LiteralPath (Join-Path $Cx.UninstallRoot $Cx.UninstallName))|lnk=$(Test-Path -LiteralPath (Join-Path $Cx.StartMenu 'Clean Voice\Clean Voice.lnk'))|run=$([bool](Get-ItemProperty -Path $Cx.RunKey -Name 'Clean Voice' -ErrorAction SilentlyContinue))"
}
""".replace("__UNINSTALL__", str(UNINSTALL))


def test_uninstall_removes_everything_and_the_rerun_says_already_removed(ps) -> None:
    out = ps(
        UNINSTALL_PRELUDE
        + r"""
$cx = New-UCx
Install-Fake $cx ('"' + $cx.Folder + '\Clean Voice.exe" --minimized')
Invoke-CvUninstall $cx
Show-U $cx
Invoke-CvUninstall $cx
"""
    )
    assert out == [
        "Clean Voice was removed. Your settings and recordings were kept.",
        "U|folder=False|key=False|lnk=False|run=False",
        "Clean Voice is already removed.",
    ]


def test_uninstall_keeps_a_run_value_that_points_elsewhere(ps) -> None:
    out = ps(
        UNINSTALL_PRELUDE
        + r"""
$cx = New-UCx
Install-Fake $cx '"C:\Program Files\Clean Voice\Clean Voice.exe" --minimized'
Invoke-CvUninstall $cx
Show-U $cx
"""
    )
    assert out[-1] == "U|folder=False|key=False|lnk=False|run=True"


def test_uninstall_refuses_while_clean_voice_runs(ps) -> None:
    out = ps(
        UNINSTALL_PRELUDE
        + r"""
$cx = New-UCx
Install-Fake $cx $null
$global:Running = 'x'
try { Invoke-CvUninstall $cx } catch { "ERR|$($_.Exception.Message)" }
Show-U $cx
"""
    )
    assert out == [
        "ERR|CVPLAIN:Clean Voice is open. Close it, then run Uninstall again.",
        "U|folder=True|key=True|lnk=True|run=False",
    ]
