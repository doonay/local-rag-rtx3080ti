param(
    [Parameter(Mandatory = $true)]
    [string]$WorkspaceRoot
)

$resolvedWorkspaceRoot = (Resolve-Path -LiteralPath $WorkspaceRoot).Path
$localRoot = Join-Path $resolvedWorkspaceRoot '.local'
$cacheRoot = Join-Path $localRoot 'cache'
$huggingFaceRoot = Join-Path $cacheRoot 'huggingface'
$tempRoot = Join-Path $localRoot 'tmp'

$localPaths = [ordered]@{
    RAG_PROJECT_LOCAL_ROOT       = $localRoot
    LOCAL_DATA_DIR               = $localRoot
    HF_HOME                      = $huggingFaceRoot
    HF_HUB_CACHE                 = (Join-Path $huggingFaceRoot 'hub')
    HUGGINGFACE_HUB_CACHE        = (Join-Path $huggingFaceRoot 'hub')
    HF_XET_CACHE                 = (Join-Path $huggingFaceRoot 'xet')
    HF_ASSETS_CACHE              = (Join-Path $huggingFaceRoot 'assets')
    HF_DATASETS_CACHE            = (Join-Path $huggingFaceRoot 'datasets')
    TRANSFORMERS_CACHE           = (Join-Path $huggingFaceRoot 'hub')
    SENTENCE_TRANSFORMERS_HOME   = (Join-Path $cacheRoot 'sentence-transformers')
    TORCH_HOME                   = (Join-Path $cacheRoot 'torch')
    XDG_CACHE_HOME               = $cacheRoot
    XDG_CONFIG_HOME              = (Join-Path $localRoot 'config')
    XDG_DATA_HOME                = (Join-Path $localRoot 'share')
    XDG_STATE_HOME               = (Join-Path $localRoot 'state')
    PIP_CACHE_DIR                = (Join-Path $cacheRoot 'pip')
    UV_CACHE_DIR                 = (Join-Path $cacheRoot 'uv')
    MPLCONFIGDIR                 = (Join-Path $cacheRoot 'matplotlib')
    NUMBA_CACHE_DIR              = (Join-Path $cacheRoot 'numba')
    TRITON_CACHE_DIR             = (Join-Path $cacheRoot 'triton')
    TORCHINDUCTOR_CACHE_DIR      = (Join-Path $cacheRoot 'torchinductor')
    CUDA_CACHE_PATH              = (Join-Path $cacheRoot 'nvidia')
    PYTHONPYCACHEPREFIX          = (Join-Path $cacheRoot 'python')
    TEMP                         = $tempRoot
    TMP                          = $tempRoot
    TMPDIR                       = $tempRoot
}

foreach ($entry in $localPaths.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    New-Item -ItemType Directory -Force -Path $entry.Value | Out-Null
}

# Make sitecustomize.py visible during Python startup, before third-party imports.
$existingPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
$pythonPathEntries = @($resolvedWorkspaceRoot)
if ($existingPythonPath) {
    $pythonPathEntries += $existingPythonPath
}
[Environment]::SetEnvironmentVariable(
    'PYTHONPATH',
    ($pythonPathEntries -join [IO.Path]::PathSeparator),
    'Process'
)
