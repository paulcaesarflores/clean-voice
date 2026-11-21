# Clean Voice uninstaller for Windows (Windows PowerShell 5.1). Public, MIT, ASCII only.
#
# install.ps1 copies this file to "<install folder>\Uninstall Clean Voice.ps1" and registers it as the UninstallString
# of Clean Voice's entry in Settings > Apps > Installed apps. It removes the entry, the Start menu shortcut, the
# "Start with Windows" value (only if it points into the install folder) and the install folder. Settings and
# recordings live outside the folder and stay, as they did with the MSI.
# No "exit": errors are thrown and printed plainly. The tests dot-source this file with CLEAN_VOICE_TEST_IMPORT set.

function New-CvUninstallContext {
    param([hashtable]$Override = @{})
    $cx = @{
        Folder        = Join-Path $env:LOCALAPPDATA 'Programs\Clean Voice'
        UninstallRoot = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall'
        UninstallName = 'CleanVoice'
        RunKey        = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
        StartMenu     = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
    }
    foreach ($key in $Override.Keys) { $cx[$key] = $Override[$key] }
    $cx
}

# Seam: the running Clean Voice processes.
function Get-CvProcess {
    Get-Process -Name 'Clean Voice' -ErrorAction SilentlyContinue
}

# Seam: the window is the guest's own (started from Installed apps) and would vanish before they read it.
function Wait-CvEnter {
    # No console to read from (a CI run, a closed window): nothing to wait for.
    try { [void](Read-Host 'Press Enter to close') } catch { $null = $_ }
}

function Invoke-CvUninstall {
    param([hashtable]$Cx)
    if (Get-CvProcess) { throw 'CVPLAIN:Clean Voice is open. Close it, then run Uninstall again.' }
    $key = Join-Path $Cx.UninstallRoot $Cx.UninstallName
    $menu = Join-Path $Cx.StartMenu 'Clean Voice'
    $link = Join-Path $menu 'Clean Voice.lnk'
    $present = (Test-Path -LiteralPath $key) -or (Test-Path -LiteralPath $link) -or (Test-Path -LiteralPath $Cx.Folder)
    if (-not $present) {
        Write-Host 'Clean Voice is already removed.'
        return
    }
    Remove-Item -LiteralPath $key -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $menu -Force -ErrorAction SilentlyContinue
    $run = Get-ItemProperty -Path $Cx.RunKey -Name 'Clean Voice' -ErrorAction SilentlyContinue
    if ($null -ne $run -and ([string]$run.'Clean Voice').ToLower().Contains($Cx.Folder.ToLower())) {
        Remove-ItemProperty -Path $Cx.RunKey -Name 'Clean Voice' -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $Cx.Folder) { Remove-Item -LiteralPath $Cx.Folder -Recurse -Force }
    Write-Host 'Clean Voice was removed. Your settings and recordings were kept.'
}

# This script lives inside the folder it deletes, so it first runs a copy of itself from %TEMP%.
function Invoke-CvUninstallMain {
    $ErrorActionPreference = 'Stop'
    $cx = New-CvUninstallContext
    try {
        $self = $PSCommandPath
        $inside = $self -and $self.ToLower().StartsWith($cx.Folder.ToLower())
        if ($inside) {
            $copy = Join-Path $env:TEMP 'Uninstall Clean Voice.ps1'
            Copy-Item -LiteralPath $self -Destination $copy -Force
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $copy
            return
        }
        Invoke-CvUninstall $cx
    } catch {
        $text = [string]$_.Exception.Message
        if ($text.StartsWith('CVPLAIN:')) { Write-Host $text.Substring(8) } else { Write-Host "Something went wrong: $text" }
    }
    Wait-CvEnter
}

if (-not $env:CLEAN_VOICE_TEST_IMPORT) { Invoke-CvUninstallMain }
