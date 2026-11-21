# Clean Voice installer for Windows 10/11 (Windows PowerShell 5.1). Public, MIT, ASCII only.
#
#   irm https://github.com/paulcaesarflores/clean-voice/releases/latest/download/install.ps1 | iex
#
# It gets its own Python (uv's managed one), fetches this release's source, checks it, builds Clean Voice with
# Briefcase (create + build) and installs it for the current user, no administrator needed (an older MSI copy that was
# installed for all users is the one case Windows asks for it). Design and sources: B019 in the project's plan.
#
# Rules kept here: one Main function called on the last line (a half-downloaded script runs nothing); no "exit"
# (under iex it would close the guest's terminal): errors are thrown, caught at the top and printed plainly.
# The tests dot-source this file with CLEAN_VOICE_TEST_IMPORT set, so it defines its functions and runs nothing.

# Source: uv's release page (.sha256 files) and Git for Windows' release notes for the two archives; the Briefcase
# pin equals pyproject.toml's dev group. Stamped by release_assets.py: the version and the source zip's SHA-256.
function Get-CvPins {
    @{
        Version       = '@@VERSION@@'
        SourceSha256  = '@@SOURCE_SHA256@@'
        Briefcase     = '0.4.7'
        UvVersion     = '0.13.9'
        UvSha256      = 'edd61f7f5728a56cc62455386661860bdfc7bab65872f39b0c25bdcc1d8e1beb'
        MinGitVersion = '2.56.1'
        MinGitTag     = 'v2.56.1.windows.1'
        MinGitSha256  = '4a92850b6e917da4fd010669b614b55dccf5b0ec333a534cc4be4700dfefdafd'
        BaseUrl       = 'https://github.com/paulcaesarflores/clean-voice/releases/latest/download'
    }
}

# The setup context: every path and setting in one table, so the tests can point each at a scratch place.
# CLEAN_VOICE_BASE and CLEAN_VOICE_TEST_IMPORT are the two test hooks; CLEAN_VOICE_UPDATE, _WAIT_PID and _STATUS_FILE
# are set by the app's Install update (src/clean_voice/updates.py).
function New-CvContext {
    param([hashtable]$Override = @{})
    $pins = Get-CvPins
    $local = $env:LOCALAPPDATA
    $uninstallPath = 'Software\Microsoft\Windows\CurrentVersion\Uninstall'
    $waitPid = 0
    if ($env:CLEAN_VOICE_WAIT_PID) { [void][int]::TryParse($env:CLEAN_VOICE_WAIT_PID, [ref]$waitPid) }
    $base = $pins.BaseUrl
    if ($env:CLEAN_VOICE_BASE) { $base = $env:CLEAN_VOICE_BASE.TrimEnd('/') }
    $cx = @{
        Pins          = $pins
        Base          = $base
        Work          = Join-Path $local 'CVSetup'
        Folder        = Join-Path $local 'Programs\Clean Voice'
        Log           = Join-Path $env:TEMP 'clean-voice-install.log'
        UninstallRoot = "HKCU:\$uninstallPath"
        UninstallName = 'CleanVoice'
        RunKey        = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
        StartMenu     = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
        OldMsiRoots      = @(
            @{ Path = "HKCU:\$uninstallPath"; Scope = 'User' },
            @{ Path = "HKLM:\$uninstallPath"; Scope = 'Machine' },
            @{ Path = "HKLM:\Software\WOW6432Node\$uninstallPath"; Scope = 'Machine' }
        )
        SacPath       = 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy'
        Update        = [bool]$env:CLEAN_VOICE_UPDATE
        WaitPid       = $waitPid
        StatusFile    = $env:CLEAN_VOICE_STATUS_FILE
        WaitSeconds   = 60
        Step          = 0
        StepName      = ''
    }
    foreach ($key in $Override.Keys) { $cx[$key] = $Override[$key] }
    $cx.Old = $cx.Folder + '.old'
    $cx.Source = Join-Path $cx.Work 'source'
    $cx.NewTree = Join-Path $cx.Source 'build\clean-voice\windows\app\src'
    foreach ($key in $Override.Keys) { $cx[$key] = $Override[$key] }
    $cx
}

# ---- Output, log, errors ---------------------------------------------------------------------------------------

function Write-CvLog {
    param([hashtable]$Cx, [string]$Text)
    try { Add-Content -LiteralPath $Cx.Log -Value $Text } catch { $null = $_ }
}

function Write-CvLine {
    param([hashtable]$Cx, [string]$Text)
    Write-Host $Text
    Write-CvLog $Cx $Text
}

function Write-CvStep {
    param([hashtable]$Cx, [int]$Number, [string]$Name)
    $Cx.Step = $Number
    $Cx.StepName = $Name
    Write-CvLine $Cx "Step $Number of 8: $Name"
}

# A plain, guest-readable failure: Stop-Install throws it, the top of Invoke-CvInstall prints it as it is. Anything
# else that is thrown gets the generic "Something went wrong at step N" line, with the real error in the log.
function Stop-Install {
    param([string]$Message)
    throw ('CVPLAIN:' + $Message)
}

function Get-CvPlainMessage {
    param($ErrorRecord)
    $text = [string]$ErrorRecord.Exception.Message
    if ($text.StartsWith('CVPLAIN:')) { return $text.Substring(8) }
    return $null
}

# The one-word status file Install update reads once a second ("swap" or "failed"); no OBS source: the user approved it.
function Write-CvStatus {
    param([hashtable]$Cx, [string]$Word)
    if (-not $Cx.StatusFile) { return }
    try { [System.IO.File]::WriteAllText($Cx.StatusFile, $Word) } catch { Write-CvLog $Cx "Could not write the status file: $_" }
}

# Seam: update mode keeps the console open to read a failure (the window is new, so it would vanish).
function Wait-CvEnter {
    # No console to read from (a CI run, a closed window): nothing to wait for.
    try { [void](Read-Host 'Press Enter to close') } catch { $null = $_ }
}

# ---- Checks ----------------------------------------------------------------------------------------------------

# Source: briefcase/platforms/windows/__init__.py:106-157 (verify_host: x86-64 or ARM64 Python; Windows 10 is build 10240).
function Get-CvPlatformProblem {
    param([string]$Arch, [int]$Build)
    if ($Arch -ne 'AMD64' -or $Build -lt 10240) {
        return 'Clean Voice needs 64-bit Windows 10 or 11 (Intel/AMD).'
    }
    return $null
}

function Get-CvArch {
    if ($env:PROCESSOR_ARCHITEW6432) { return $env:PROCESSOR_ARCHITEW6432 }
    return $env:PROCESSOR_ARCHITECTURE
}

# Seam: the running Clean Voice processes.
function Get-CvProcess {
    Get-Process -Name 'Clean Voice' -ErrorAction SilentlyContinue
}

# Source: Windows' own Smart App Control state (registry value VerifiedAndReputablePolicyState; 0 = off), user
# decision 2026-11-21: warn and install anyway (DESIGN, Packaging).
function Get-CvSacState {
    param([hashtable]$Cx)
    $item = Get-ItemProperty -Path $Cx.SacPath -Name 'VerifiedAndReputablePolicyState' -ErrorAction SilentlyContinue
    if ($null -eq $item) { return $null }
    return $item.VerifiedAndReputablePolicyState
}

function Show-CvSacWarning {
    param([hashtable]$Cx, $State)
    if ($null -eq $State -or [int]$State -eq 0) { return }
    Write-CvLine $Cx ("Windows' Smart App Control is on and may block Clean Voice, because we can't sign it. " +
        "To turn it off: Windows Security > App & browser control > Smart App Control settings > Off " +
        "(this cannot be turned on again without resetting Windows). Installing anyway.")
}

# ---- Downloads -------------------------------------------------------------------------------------------------

function Get-CvFile {
    param([hashtable]$Cx, [string]$Url, [string]$Dest)
    Write-CvLog $Cx "Downloading $Url"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
    } catch {
        Write-CvLog $Cx "Download failed: $_"
        Stop-Install "Couldn't reach GitHub. Check your internet connection and run the command again."
    }
}

function Assert-CvHash {
    param([string]$Path, [string]$Expected)
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
    if ($actual -ne $Expected) {
        Stop-Install "The download didn't match what was published, so nothing was installed."
    }
}

# Source: .NET's System.IO.Compression.ZipFile (Windows PowerShell 5.1 has no Expand-Archive speed or long-path care).
function Expand-CvZip {
    param([string]$Zip, [string]$Dest)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($Zip, $Dest)
}

function Remove-CvTree {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force }
}

function Invoke-CvNative {
    param([hashtable]$Cx, [string]$File, [string[]]$Arguments)
    Write-CvLog $Cx ("> " + $File + ' ' + ($Arguments -join ' '))
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $File @Arguments 2>&1 | ForEach-Object { Write-CvLog $Cx ([string]$_) }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $saved
    }
    if ($code -ne 0) { throw "$File exited with code $code" }
}

# ---- Environment for the script's own process only ---------------------------------------------------------------

function Save-CvEnv {
    $saved = @{}
    foreach ($name in 'Path', 'UV_PYTHON_PREFERENCE', 'UV_PYTHON_INSTALL_DIR', 'UV_CACHE_DIR', 'BRIEFCASE_HOME') {
        $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    }
    $saved
}

function Restore-CvEnv {
    param([hashtable]$Saved)
    foreach ($name in $Saved.Keys) { [Environment]::SetEnvironmentVariable($name, $Saved[$name], 'Process') }
}

# ---- Old MSI ----------------------------------------------------------------------------------------------------

# Source: the Briefcase windows template's .wxs (read from `briefcase create windows`): Manufacturer and Name
# "Clean Voice", Scope perUserOrMachine, default WixPerUserFolder. An MSI product's Uninstall key is its GUID and
# carries WindowsInstaller = 1; HKCU holds a per-user install, HKLM (64 and 32-bit views) a per-machine one.
function Find-CvOldMsi {
    param([hashtable]$Cx)
    $found = @{ User = @(); Machine = @() }
    foreach ($root in $Cx.OldMsiRoots) {
        if (-not (Test-Path -LiteralPath $root.Path)) { continue }
        foreach ($key in Get-ChildItem -LiteralPath $root.Path) {
            if ($key.PSChildName -notmatch '^\{[0-9A-Fa-f-]{36}\}$') { continue }
            $props = Get-ItemProperty -LiteralPath $key.PSPath
            if ($props.DisplayName -eq 'Clean Voice' -and $props.WindowsInstaller -eq 1) {
                $found[$root.Scope] += $key.PSChildName
            }
        }
    }
    $found
}

# Source: Microsoft's msiexec (/x = uninstall, /qn = no UI; 0 and 3010 "restart needed" are success). A per-machine
# product is removed with Start-Process -Verb RunAs, which shows Windows' own admin prompt: a plain msiexec does not
# prompt, it fails with 1925. A declined prompt throws "cancelled by the user".
function Remove-CvMsi {
    param([string]$Code, [bool]$Machine)
    $declined = ("An older Clean Voice couldn't be removed (Windows asked for an administrator and it was declined, " +
        "or it failed). Remove 'Clean Voice' in Settings > Installed apps, then run the command again.")
    $argText = "/x $Code /qn"
    try {
        if ($Machine) {
            $process = Start-Process -FilePath 'msiexec' -ArgumentList $argText -Verb RunAs -Wait -PassThru
        } else {
            $process = Start-Process -FilePath 'msiexec' -ArgumentList $argText -Wait -PassThru
        }
    } catch {
        Stop-Install $declined
    }
    if ($process.ExitCode -ne 0 -and $process.ExitCode -ne 3010) { Stop-Install $declined }
}

# ---- Register ---------------------------------------------------------------------------------------------------

# Source: Microsoft Learn, "Uninstall registry key" (the per-user form of the key Installed apps lists), and
# WScript.Shell CreateShortcut for the Start menu entry. The Run value: only the one case where the exe path changes
# (an old per-machine copy was removed) rewrites it; a per-user MSI's value already names this very folder.
function Register-CvInstall {
    param([hashtable]$Cx, [string]$Version, [bool]$RewriteRun = $false)
    $exe = Join-Path $Cx.Folder 'Clean Voice.exe'
    $script = Join-Path $Cx.Folder 'Uninstall Clean Voice.ps1'
    Copy-Item -LiteralPath (Join-Path $Cx.Source 'tools\install\uninstall.ps1') -Destination $script -Force
    $command = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + $script + '"'
    $key = Join-Path $Cx.UninstallRoot $Cx.UninstallName
    New-Item -Path $key -Force | Out-Null
    $strings = [ordered]@{
        DisplayName     = 'Clean Voice'
        DisplayVersion  = $Version
        Publisher       = 'Clean Voice'
        DisplayIcon     = $exe
        InstallLocation = $Cx.Folder
        UninstallString = $command
    }
    foreach ($name in $strings.Keys) {
        New-ItemProperty -Path $key -Name $name -Value $strings[$name] -PropertyType String -Force | Out-Null
    }
    foreach ($name in 'NoModify', 'NoRepair') {
        New-ItemProperty -Path $key -Name $name -Value 1 -PropertyType DWord -Force | Out-Null
    }
    # The same subfolder the old MSI used (Programs\Clean Voice\Clean Voice.lnk), user decision 2026-11-21.
    $menu = Join-Path $Cx.StartMenu 'Clean Voice'
    New-Item -ItemType Directory -Path $menu -Force | Out-Null
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut((Join-Path $menu 'Clean Voice.lnk'))
    $link.TargetPath = $exe
    $link.WorkingDirectory = $Cx.Folder
    $link.Description = 'Clean Voice'
    $link.Save()
    if ($RewriteRun) {
        $run = Get-ItemProperty -Path $Cx.RunKey -Name 'Clean Voice' -ErrorAction SilentlyContinue
        if ($null -ne $run) {
            Set-ItemProperty -Path $Cx.RunKey -Name 'Clean Voice' -Value ('"' + $exe + '" --minimized')
        }
    }
}

# Seam: start the app (update mode only).
function Start-CvApp {
    param([string]$Exe)
    Start-Process -FilePath $Exe -WorkingDirectory (Split-Path -Parent $Exe)
}

# Seam: move a folder (so the tests can make one move fail).
function Move-CvTree {
    param([string]$From, [string]$To)
    Move-Item -LiteralPath $From -Destination $To
}

# ---- Swap ---------------------------------------------------------------------------------------------------------

function Wait-CvAppExit {
    param([hashtable]$Cx)
    if (-not $Cx.WaitPid) { return }
    $deadline = (Get-Date).AddSeconds($Cx.WaitSeconds)
    while (Get-Process -Id $Cx.WaitPid -ErrorAction SilentlyContinue) {
        if ((Get-Date) -gt $deadline) {
            Stop-Install "Clean Voice didn't close, so the old version is still in place."
        }
        Start-Sleep -Milliseconds 200
    }
}

# Rollback: put the old folder back and, when this run took its registration away (a per-user MSI removal does),
# register it again; in update mode reopen the old app and report "failed". Never throws.
function Undo-CvSwap {
    param([hashtable]$Cx, [bool]$HadOld)
    try {
        if ((-not $HadOld) -or (Test-Path -LiteralPath $Cx.Old)) { Remove-CvTree $Cx.Folder }
        if ($HadOld -and (Test-Path -LiteralPath $Cx.Old)) { Move-CvTree $Cx.Old $Cx.Folder }
        if ($HadOld) {
            $version = $Cx.Pins.Version
            try { $version = (Get-Item -LiteralPath (Join-Path $Cx.Folder 'Clean Voice.exe')).VersionInfo.ProductVersion } catch { $null = $_ }
            if (-not $version) { $version = $Cx.Pins.Version }
            Register-CvInstall $Cx $version
            if ($Cx.Update) { Start-CvApp (Join-Path $Cx.Folder 'Clean Voice.exe') }
        }
    } catch {
        Write-CvLog $Cx "Rollback problem: $_"
    }
    if ($Cx.Update) { Write-CvStatus $Cx 'failed' }
}

# Seam-free helper for the log: what a folder holds, two levels deep.
function Get-CvListing {
    param([string]$Path)
    try {
        $names = Get-ChildItem -LiteralPath $Path -Recurse -Depth 1 -Force -ErrorAction Stop | ForEach-Object { $_.FullName.Substring($Path.Length) }
        return '[' + (@($names) -join ', ') + ']'
    } catch {
        return '(cannot list: ' + $_ + ')'
    }
}

# Order, so a failure leaves a working Clean Voice. The per-user MSI's files are in the very folder the new app goes
# to, so it is removed while that folder is renamed aside (an MSI removal skips files that are already gone: Microsoft
# Learn, RemoveFiles action, "removed if the file exists in the specified directory"; the migration run in CI is the
# proof for the installed files). The per-machine MSI's files are elsewhere, so it is removed only after the new app is
# in place. Rollback only, no recovery journal.
function Invoke-CvSwap {
    param([hashtable]$Cx, [hashtable]$Msi)
    if ($Cx.Update) {
        Write-CvStatus $Cx 'swap'
        Wait-CvAppExit $Cx
    }
    $hadOld = Test-Path -LiteralPath $Cx.Folder
    $machineRemoved = $false
    try {
        Remove-CvTree $Cx.Old
        if ($hadOld) { Move-CvTree $Cx.Folder $Cx.Old }
        foreach ($code in $Msi.User) { Remove-CvMsi $code $false }
        New-Item -ItemType Directory -Path (Split-Path -Parent $Cx.Folder) -Force | Out-Null
        # Whatever stands at the folder now is not ours (the old copy is in .old): a folder left there would make the
        # move put the new tree INSIDE it (CI run 44898585914: "Clean Voice.exe is missing" after a per-user MSI removal).
        if ($hadOld -and (Test-Path -LiteralPath $Cx.Folder)) {
            Write-CvLog $Cx ('Folder reappeared after the MSI removal: ' + (Get-CvListing $Cx.Folder))
            Remove-CvTree $Cx.Folder
        }
        Move-CvTree $Cx.NewTree $Cx.Folder
        if (-not (Test-Path -LiteralPath (Join-Path $Cx.Folder 'Clean Voice.exe'))) {
            throw ('Clean Voice.exe is not in the new folder: ' + (Get-CvListing $Cx.Folder))
        }
        foreach ($code in $Msi.Machine) { Remove-CvMsi $code $true; $machineRemoved = $true }
        Remove-CvTree $Cx.Old
    } catch {
        $plain = Get-CvPlainMessage $_
        Write-CvLog $Cx "Swap failed: $_"
        Undo-CvSwap $Cx $hadOld
        if ($null -ne $plain) { Stop-Install $plain }
        if ($hadOld -and $Cx.Update) { Stop-Install "Couldn't replace the old Clean Voice. Your old version is back and was reopened." }
        if ($hadOld) { Stop-Install "Couldn't replace the old Clean Voice. Your old version is back. Nothing was changed." }
        throw
    }
    return [bool]$machineRemoved
}

# ---- The install ------------------------------------------------------------------------------------------------

function Invoke-CvInstall {
    param([hashtable]$Cx = $null)
    if ($null -eq $Cx) { $Cx = New-CvContext }
    $ProgressPreference = 'SilentlyContinue'
    $ErrorActionPreference = 'Stop'
    $saved = Save-CvEnv
    $location = Get-Location
    try {
        Set-Content -LiteralPath $Cx.Log -Value ('Clean Voice install ' + $Cx.Pins.Version + ' ' + (Get-Date)) -Encoding Ascii
        $pins = $Cx.Pins

        Write-CvStep $Cx 1 'Checking your PC'
        $problem = Get-CvPlatformProblem (Get-CvArch) ([Environment]::OSVersion.Version.Build)
        if ($problem) { Stop-Install $problem }
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        if ((-not $Cx.Update) -and (Get-CvProcess)) { Stop-Install 'Clean Voice is open. Close it, then run the command again.' }
        Show-CvSacWarning $Cx (Get-CvSacState $Cx)
        Remove-CvTree $Cx.Work
        New-Item -ItemType Directory -Path $Cx.Work -Force | Out-Null

        Write-CvStep $Cx 2 'Getting the build tools'
        $uvZip = Join-Path $Cx.Work 'uv.zip'
        Get-CvFile $Cx ('https://github.com/astral-sh/uv/releases/download/' + $pins.UvVersion + '/uv-x86_64-pc-windows-msvc.zip') $uvZip
        Assert-CvHash $uvZip $pins.UvSha256
        Expand-CvZip $uvZip (Join-Path $Cx.Work 'uv')
        $gitZip = Join-Path $Cx.Work 'mingit.zip'
        Get-CvFile $Cx ('https://github.com/git-for-windows/git/releases/download/' + $pins.MinGitTag + '/MinGit-' + $pins.MinGitVersion + '-64-bit.zip') $gitZip
        Assert-CvHash $gitZip $pins.MinGitSha256
        Expand-CvZip $gitZip (Join-Path $Cx.Work 'mingit')

        Write-CvStep $Cx 3 'Getting Clean Voice'
        $sourceZip = Join-Path $Cx.Work 'source.zip'
        Get-CvFile $Cx ($Cx.Base + '/Clean-Voice-source.zip') $sourceZip
        Assert-CvHash $sourceZip $pins.SourceSha256
        Expand-CvZip $sourceZip $Cx.Source

        Write-CvStep $Cx 4 'Getting Python'
        $uv = Join-Path $Cx.Work 'uv\uv.exe'
        $venv = Join-Path $Cx.Work 'venv'
        $python = Join-Path $venv 'Scripts\python.exe'
        # Source: docs.astral.sh/uv/concepts/python-versions. 3.12 because Briefcase builds for the running Python
        # (briefcase/platforms/base.py:727-735) and pyproject.toml says >=3.12,<3.13; --seed because Briefcase's own
        # venv needs pip (briefcase/integrations/virtual_environment/venv.py:48-72).
        $env:UV_PYTHON_PREFERENCE = 'only-managed'
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $Cx.Work 'python'
        $env:UV_CACHE_DIR = Join-Path $Cx.Work 'uvcache'
        Invoke-CvNative $Cx $uv @('python', 'install', '3.12')
        Invoke-CvNative $Cx $uv @('venv', '--python', '3.12', '--seed', $venv)
        Invoke-CvNative $Cx $uv @('pip', 'install', '--python', $python, ('briefcase==' + $pins.Briefcase))

        Write-CvStep $Cx 5 'Building Clean Voice (a few minutes)'
        # Source: briefcase/commands/create.py:1060-1066 and integrations/git.py:38-76 (create needs git: MinGit, put on
        # this process's PATH only); create.py:87-113 (binary wheels only, so no compiler); default format is `app`
        # (briefcase/platforms/windows/__init__.py:29).
        $env:Path = (Join-Path $Cx.Work 'mingit\cmd') + ';' + $env:Path
        # Briefcase refuses a BRIEFCASE_HOME that does not exist (briefcase/commands/base.py:211-217), so make it first.
        $env:BRIEFCASE_HOME = (New-Item -ItemType Directory -Force -Path (Join-Path $Cx.Work 'briefcase')).FullName
        Set-Location -LiteralPath $Cx.Source
        Invoke-CvNative $Cx $python @('-m', 'briefcase', 'create', 'windows', '--no-input')
        Invoke-CvNative $Cx $python @('-m', 'briefcase', 'build', 'windows', '--no-input')
        Invoke-CvNative $Cx $python @('tools\ci\set_copyright.py', (Join-Path $Cx.NewTree 'Clean Voice.exe'))
        Set-Location -LiteralPath $location
        # The marker the app reads to know the command installed it, so Install update rebuilds through the command
        # (an MSI copy keeps the installer's update path): src/clean_voice/updates.py MARKER_NAME.
        Set-Content -LiteralPath (Join-Path $Cx.NewTree 'installed-by-command') -Value $pins.Version -Encoding Ascii

        $msi = @{ User = @(); Machine = @() }
        if (-not $Cx.Update) {
            Write-CvStep $Cx 6 'Looking for an older copy'
            $msi = Find-CvOldMsi $Cx
        }

        Write-CvStep $Cx 7 'Installing'
        $machineRemoved = Invoke-CvSwap $Cx $msi

        Write-CvStep $Cx 8 'Finishing'
        try {
            Register-CvInstall $Cx $pins.Version $machineRemoved
        } catch {
            Write-CvLog $Cx "Registration failed: $_"
            Write-CvLine $Cx "Clean Voice is installed, but Windows couldn't list it in Installed apps / the Start menu."
        }
        if ($Cx.Update) {
            Start-CvApp (Join-Path $Cx.Folder 'Clean Voice.exe')
            Write-CvLine $Cx 'Clean Voice was updated and is opening again.'
        } else {
            Write-CvLine $Cx 'Clean Voice is installed. Open it from the Start menu.'
        }
    } catch {
        $plain = Get-CvPlainMessage $_
        Write-CvLog $Cx "Error: $_"
        if ($null -ne $plain) {
            Write-CvLine $Cx $plain
        } else {
            Write-CvLine $Cx ('Something went wrong at step ' + $Cx.Step + ' (' + $Cx.StepName + '). Nothing was changed. The log is at ' + $Cx.Log + '; send it to your host.')
        }
        Write-CvStatus $Cx 'failed'
        if ($Cx.Update) { Wait-CvEnter }
    } finally {
        Set-Location -LiteralPath $location
        Restore-CvEnv $saved
        try { Remove-CvTree $Cx.Work } catch { Write-CvLog $Cx "Could not remove the work folder: $_" }
    }
}

if (-not $env:CLEAN_VOICE_TEST_IMPORT) { Invoke-CvInstall }
