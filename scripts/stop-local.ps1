$ErrorActionPreference = 'Stop'

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$localRoot = Join-Path $workspaceRoot '.local'
$expectedRoots = @(
    (Join-Path $workspaceRoot '.venv-local'),
    (Join-Path $localRoot 'llama.cpp')
)

foreach ($name in @('api','llama')) {
    $pidPath = Join-Path $localRoot "$name.pid"
    if (-not (Test-Path -LiteralPath $pidPath)) { continue }
    $processId = [int](Get-Content -LiteralPath $pidPath -Raw)
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
        $processPath = $process.Path
        $isExpected = $false
        foreach ($root in $expectedRoots) {
            if ($processPath -and $processPath.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
                $isExpected = $true
                break
            }
        }
        if ($isExpected) {
            Stop-Process -Id $processId -Force
            Write-Host "Stopped $name (PID $processId)."
        } else {
            Write-Warning "PID $processId no longer belongs to Local RAG; it was not stopped."
        }
    }
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
}

Write-Host 'Local RAG is stopped.' -ForegroundColor Green
