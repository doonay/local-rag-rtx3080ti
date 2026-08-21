$ErrorActionPreference = 'Stop'

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'use-local-storage.ps1') -WorkspaceRoot $workspaceRoot
$localRoot = Join-Path $workspaceRoot '.local'
$venvRoot = Join-Path $workspaceRoot '.venv-local'
$llamaRoot = Join-Path $localRoot 'llama.cpp'
$modelRoot = Join-Path $localRoot 'models'
$downloadRoot = Join-Path $localRoot 'downloads'
$modelPath = Join-Path $modelRoot 'Qwen3-8B-Q4_K_M.gguf'
$markerPath = Join-Path $localRoot 'setup-complete.txt'

New-Item -ItemType Directory -Force -Path $localRoot,$llamaRoot,$modelRoot,$downloadRoot,(Join-Path $localRoot 'logs') | Out-Null
Set-Location $workspaceRoot

Write-Host "Project storage: $localRoot"
Write-Host "Download/cache storage: $env:HF_HUB_CACHE"
Write-Host 'Checking Python 3.11+...'
$pythonCommand = Get-Command python -ErrorAction Stop
$pythonVersion = & $pythonCommand.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$pythonVersion -lt [version]'3.11') {
    throw "Python 3.11 or newer is required. Found $pythonVersion."
}

$venvPython = Join-Path $venvRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host 'Creating isolated Python environment...'
    & $pythonCommand.Source -m venv $venvRoot
}

# Keep direct invocations of this project's virtualenv isolated too, even when
# they are launched without start-local.bat.
$sitePackages = & $venvPython -c "import site; print(site.getsitepackages()[0])"
$storagePth = Join-Path $sitePackages 'local-rag-storage.pth'
$storagePthContent = @(
    $workspaceRoot
    'import common.local_storage as _local_storage; _local_storage.configure_local_storage()'
) -join "`r`n"
Set-Content -LiteralPath $storagePth -Value $storagePthContent -Encoding ASCII

Write-Host 'Installing Python dependencies. The first run can take a long time...'
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
& $venvPython -m pip install -r (Join-Path $workspaceRoot 'requirements-local.txt')

function Invoke-ResumableDownload {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$Destination
    )
    if (Test-Path -LiteralPath $Destination) {
        Write-Host "Resuming/checking $(Split-Path -Leaf $Destination)..."
    } else {
        Write-Host "Downloading $(Split-Path -Leaf $Destination)..."
    }
    & curl.exe -L --fail --retry 5 --retry-delay 3 --continue-at - --output $Destination $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Url"
    }
}

$llamaServer = Get-ChildItem -LiteralPath $llamaRoot -Filter 'llama-server.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $llamaServer) {
    Write-Host 'Downloading the current official llama.cpp CUDA 12.4 build...'
    $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' -Headers @{'User-Agent'='Local-RAG-setup'}
    $mainAsset = $release.assets | Where-Object { $_.name -match '^llama-.*-bin-win-cuda-12\.4-x64\.zip$' } | Select-Object -First 1
    $cudaAsset = $release.assets | Where-Object { $_.name -eq 'cudart-llama-bin-win-cuda-12.4-x64.zip' } | Select-Object -First 1
    if (-not $mainAsset -or -not $cudaAsset) {
        throw 'Could not find the official llama.cpp CUDA 12.4 Windows assets.'
    }
    foreach ($asset in @($mainAsset, $cudaAsset)) {
        $archivePath = Join-Path $downloadRoot $asset.name
        Invoke-ResumableDownload -Url $asset.browser_download_url -Destination $archivePath
        Expand-Archive -LiteralPath $archivePath -DestinationPath $llamaRoot -Force
    }
}

if (-not (Test-Path -LiteralPath $modelPath)) {
    Write-Host 'Downloading Qwen3-8B Q4_K_M GGUF (several gigabytes)...'
    Invoke-ResumableDownload `
        -Url 'https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf?download=true' `
        -Destination $modelPath
}
if ((Get-Item -LiteralPath $modelPath).Length -lt 1GB) {
    throw 'The downloaded GGUF model is unexpectedly small. Delete it and retry setup.'
}

Set-Content -LiteralPath $markerPath -Value "Completed $(Get-Date -Format o)" -Encoding UTF8
Write-Host ''
Write-Host 'Local setup is complete. Run start-local.bat.' -ForegroundColor Green
