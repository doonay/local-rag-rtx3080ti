param(
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'use-local-storage.ps1') -WorkspaceRoot $workspaceRoot
$localRoot = Join-Path $workspaceRoot '.local'
$venvPython = Join-Path $workspaceRoot '.venv-local\Scripts\python.exe'
$modelPath = Join-Path $localRoot 'models\Qwen3-8B-Q4_K_M.gguf'
$logsRoot = Join-Path $localRoot 'logs'
$markerPath = Join-Path $localRoot 'setup-complete.txt'

function Test-HttpReady {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $markerPath) -or -not (Test-Path -LiteralPath $venvPython) -or -not (Test-Path -LiteralPath $modelPath)) {
    Write-Host 'First launch: preparing the local environment...'
    & (Join-Path $PSScriptRoot 'setup-local.ps1')
}

$llamaServer = Get-ChildItem -LiteralPath (Join-Path $localRoot 'llama.cpp') -Filter 'llama-server.exe' -Recurse | Select-Object -First 1
if (-not $llamaServer) {
    throw 'llama-server.exe is missing. Run setup-local.bat again.'
}
New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null

Write-Host "Project storage: $localRoot"
Write-Host "Model cache: $env:HF_HUB_CACHE"
$env:EMBEDDING_MODEL = 'BAAI/bge-m3'
$env:EMBEDDING_DEVICE = 'cpu'
$env:EMBEDDING_DIMENSION = '1024'
$env:EMBEDDING_BATCH_SIZE = '8'
$env:EMBEDDING_MAX_LENGTH = '1024'
$env:RERANKER_MODEL = 'Qwen/Qwen3-Reranker-0.6B'
$env:RERANKER_BATCH_SIZE = '4'
$env:RERANKER_MAX_LENGTH = '2048'
$env:RERANKER_USE_FP16 = 'true'
$env:MAX_UPLOAD_BYTES = '536870912'
$env:LLAMA_URL = 'http://127.0.0.1:8001'
$env:LLM_MODEL = 'Qwen3-8B-Q4_K_M.gguf'
$env:TOKENIZERS_PARALLELISM = 'false'

$llamaStartedHere = $false
if (-not (Test-HttpReady 'http://127.0.0.1:8001/health')) {
    Write-Host 'Starting Qwen3-8B with llama.cpp on RTX 3080 Ti...'
    $llamaProcess = Start-Process `
        -FilePath $llamaServer.FullName `
        -ArgumentList @('-m',$modelPath,'-ngl','99','-c','8192','-np','1','--host','127.0.0.1','--port','8001') `
        -WorkingDirectory $workspaceRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logsRoot 'llama.stdout.log') `
        -RedirectStandardError (Join-Path $logsRoot 'llama.stderr.log') `
        -PassThru
    Set-Content -LiteralPath (Join-Path $localRoot 'llama.pid') -Value $llamaProcess.Id
    $llamaStartedHere = $true
    for ($attempt = 0; $attempt -lt 180; $attempt++) {
        if (Test-HttpReady 'http://127.0.0.1:8001/health') { break }
        if ($llamaProcess.HasExited) { throw 'llama.cpp stopped during startup. See .local\logs\llama.stderr.log.' }
        Start-Sleep -Seconds 1
    }
    if (-not (Test-HttpReady 'http://127.0.0.1:8001/health')) {
        Stop-Process -Id $llamaProcess.Id -Force -ErrorAction SilentlyContinue
        throw 'llama.cpp did not become ready within three minutes.'
    }
}

if (-not (Test-HttpReady 'http://127.0.0.1:8000/health')) {
    Write-Host 'Loading BGE-M3 on CPU and Qwen3 Reranker on GPU...'
    $apiProcess = Start-Process `
        -FilePath $venvPython `
        -ArgumentList @('-m','uvicorn','local_app.main:app','--host','127.0.0.1','--port','8000') `
        -WorkingDirectory $workspaceRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logsRoot 'api.stdout.log') `
        -RedirectStandardError (Join-Path $logsRoot 'api.stderr.log') `
        -PassThru
    Set-Content -LiteralPath (Join-Path $localRoot 'api.pid') -Value $apiProcess.Id
    for ($attempt = 0; $attempt -lt 900; $attempt++) {
        if (Test-HttpReady 'http://127.0.0.1:8000/health') { break }
        if ($apiProcess.HasExited) {
            if ($llamaStartedHere) { Stop-Process -Id $llamaProcess.Id -Force -ErrorAction SilentlyContinue }
            throw 'Local API stopped during startup. See .local\logs\api.stderr.log.'
        }
        if ($attempt -gt 0 -and $attempt % 10 -eq 0) { Write-Host 'Models are still loading...' }
        Start-Sleep -Seconds 1
    }
    if (-not (Test-HttpReady 'http://127.0.0.1:8000/health')) {
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
        if ($llamaStartedHere) { Stop-Process -Id $llamaProcess.Id -Force -ErrorAction SilentlyContinue }
        throw 'Local API did not become ready within fifteen minutes.'
    }
}

Write-Host ''
Write-Host 'Local RAG is ready: http://127.0.0.1:8000' -ForegroundColor Green
Write-Host 'Use stop-local.bat to stop it.'
if (-not $NoBrowser) {
    Start-Process 'http://127.0.0.1:8000'
}
