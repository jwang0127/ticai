param(
    [string]$Repository = "",
    [string]$Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Repository)) {
    $Repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$logDir = Join-Path $Repository "output\automation"
$worktree = Join-Path $Repository ".automation-worktree"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logDir "refresh-$stamp.log"
$outputPath = Join-Path $logDir "refresh-$stamp.out"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-LoggedStep {
    param([string]$File, [string[]]$Arguments, [string]$WorkingDirectory)
    Write-Host "[RUN] $File $($Arguments -join ' ')"
    Write-Host "[CWD] $WorkingDirectory (git=$([bool](Test-Path (Join-Path $WorkingDirectory '.git'))))"
    Push-Location $WorkingDirectory
    try {
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $File @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorAction
        if ($exitCode -ne 0) { throw "Command failed with exit code ${exitCode}: $File" }
    }
    finally {
        Pop-Location
    }
}

try {
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Python executable not found: $Python"
    }

    Invoke-LoggedStep "git" @("fetch", "origin", "main") -WorkingDirectory $Repository

    if (Test-Path -LiteralPath $worktree) {
        Invoke-LoggedStep "git" @("worktree", "remove", "--force", $worktree) -WorkingDirectory $Repository
    }
    Invoke-LoggedStep "git" @("worktree", "add", "--detach", $worktree, "origin/main") -WorkingDirectory $Repository

    $env:GITHUB_OUTPUT = $outputPath
    $refreshCode = 1
    $games = @()
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        if (Test-Path -LiteralPath $outputPath) { Remove-Item -LiteralPath $outputPath -Force }
        Write-Host "[REFRESH] attempt $attempt/5"
        $ErrorActionPreference = "Continue"
        & $Python (Join-Path $worktree "src\refresh_scheduled.py") --games due --history-limit 2000 2>&1 |
            Tee-Object -FilePath $logPath -Append
        $refreshCode = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        $games = @()
        if (Test-Path -LiteralPath $outputPath) {
            $line = Get-Content -LiteralPath $outputPath | Where-Object { $_ -like "games=*" } | Select-Object -Last 1
            if ($line) { $games = @($line.Substring(6).Split(',') | Where-Object { $_ }) }
        }
        if ($refreshCode -eq 0 -and $games.Count -gt 0) { break }
        if ($attempt -lt 5) {
            Write-Host "[WAIT] no verified fresh games; retrying in 15 minutes"
            Start-Sleep -Seconds 900
        }
    }

    if ($refreshCode -ne 0 -or $games.Count -eq 0) {
        Write-Host "[WAIT] no verified fresh games; keep cache and retry next trigger"
        exit 0
    }

    $gameArg = $games -join ','
    Invoke-LoggedStep $Python @("src\optimize_models.py", "--games", $gameArg) -WorkingDirectory $worktree
    Invoke-LoggedStep $Python @("src\generate_dashboard.py", "--games", $gameArg) -WorkingDirectory $worktree
    Invoke-LoggedStep $Python @("-m", "unittest", "discover", "-s", "tests", "-v") -WorkingDirectory $worktree

    Invoke-LoggedStep "git" @("add", "data/processed/draws.json", "data/processed/model_reviews.json", "data/processed/model_tuning.json", "docs/assets/data/dashboard.json") -WorkingDirectory $worktree
    Push-Location $worktree
    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] no data changes"
        exit 0
    }
    Invoke-LoggedStep "git" @("config", "user.name", "ticai-local-refresh") -WorkingDirectory $worktree
    Invoke-LoggedStep "git" @("config", "user.email", "ticai-local-refresh@users.noreply.github.com") -WorkingDirectory $worktree
    Invoke-LoggedStep "git" @("commit", "-m", "chore(data): refresh lottery dashboard") -WorkingDirectory $worktree
    Invoke-LoggedStep "git" @("push", "origin", "HEAD:main") -WorkingDirectory $worktree
    Write-Host "[OK] pushed verified refresh for $gameArg"
}
finally {
    Pop-Location -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $worktree) {
        Push-Location $Repository
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & git worktree remove --force $worktree 2>&1 | Tee-Object -FilePath $logPath -Append
        $ErrorActionPreference = $previousErrorAction
        Pop-Location
    }
}
