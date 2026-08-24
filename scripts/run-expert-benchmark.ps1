[CmdletBinding()]
param(
    [string] $BinDir = "D:\llama-wackMall\llama-wackMall-main\build-msvc-vulkan-x64\bin",
    [string] $Model = "D:\model\Qwen3.5-122B-A10B-UD-IQ2_M.gguf",
    [string] $ResultsDir = "",
    [double] $RamPoolGb = 1,
    [int] $SampleIntervalMilliseconds = 1000,
    [string[]] $Only = @()
)

$ErrorActionPreference = "Stop"

$llamaCli = Join-Path $BinDir "llama-cli.exe"
if (-not (Test-Path -LiteralPath $llamaCli)) {
    throw "llama-cli.exe not found: $llamaCli"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found: $Model"
}
if ($RamPoolGb -le 0) {
    throw "RamPoolGb must be greater than zero because the PREAD experiment requires LLAMA_EXPERT_RAMPOOL."
}
if ($SampleIntervalMilliseconds -lt 100) {
    throw "SampleIntervalMilliseconds must be at least 100."
}

if ([string]::IsNullOrWhiteSpace($ResultsDir)) {
    $ResultsDir = Join-Path $PSScriptRoot "..\expert-benchmark-results"
}
$ResultsDir = [System.IO.Path]::GetFullPath($ResultsDir)
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$gpuCounterPaths = @(
    "\GPU Adapter Memory(*)\Shared Usage",
    "\GPU Adapter Memory(*)\Dedicated Usage",
    "\GPU Local Adapter Memory(*)\Local Usage"
)
$gpuCountersAvailable = $false
try {
    Get-Counter -Counter $gpuCounterPaths -ErrorAction Stop | Out-Null
    $gpuCountersAvailable = $true
} catch {
    Write-Warning "Windows GPU memory counters are unavailable. GPU memory fields will be empty."
}

$computerInfo = $null
try {
    Add-Type -AssemblyName Microsoft.VisualBasic -ErrorAction Stop
    $computerInfo = New-Object Microsoft.VisualBasic.Devices.ComputerInfo
} catch {
    Write-Warning "Physical-memory counters are unavailable. Only process memory will be recorded."
}

function Set-ExpertEnvironment {
    param(
        [hashtable] $Settings,
        [string] $StatsPath,
        [string] $PredictLogPath
    )

    $defaults = @{
        LLAMA_EXPERT_S                = "180"
        LLAMA_EXPERT_ADAPT            = "0"
        LLAMA_EXPERT_PREDICT          = "0"
        LLAMA_EXPERT_PREFETCH_GB      = "0"
        LLAMA_EXPERT_PREFETCH_THREADS = "0"
        LLAMA_EXPERT_PREFETCH_MB      = "32"
        LLAMA_EXPERT_RAMPOOL          = "0"
        LLAMA_EXPERT_MADVISE          = "1"
        LLAMA_EXPERT_PREAD            = "0"
        LLAMA_EXPERT_STATS            = $StatsPath
    }

    foreach ($name in $defaults.Keys) {
        [Environment]::SetEnvironmentVariable($name, $defaults[$name], "Process")
    }
    [Environment]::SetEnvironmentVariable("LLAMA_EXPERT_PREDICT_LOG", $null, "Process")

    foreach ($name in $Settings.Keys) {
        [Environment]::SetEnvironmentVariable($name, [string] $Settings[$name], "Process")
    }
    if ($PredictLogPath) {
        [Environment]::SetEnvironmentVariable("LLAMA_EXPERT_PREDICT_LOG", $PredictLogPath, "Process")
    }
}

function Get-ResourceSample {
    param([System.Diagnostics.Process] $Process)

    $totalRamMiB = $null
    $freeRamMiB = $null
    if ($computerInfo) {
        $totalRamMiB = [math]::Round($computerInfo.TotalPhysicalMemory / 1MB, 1)
        $freeRamMiB = [math]::Round($computerInfo.AvailablePhysicalMemory / 1MB, 1)
    }

    $sample = [ordered]@{
        timestamp_utc       = [DateTime]::UtcNow.ToString("o")
        system_ram_used_mib = if ($null -ne $totalRamMiB) { [math]::Round($totalRamMiB - $freeRamMiB, 1) } else { $null }
        system_ram_total_mib = $totalRamMiB
        system_ram_available_mib = $freeRamMiB
        process_working_set_mib = $null
        process_private_mib = $null
        gpu = @()
    }

    try {
        $Process.Refresh()
        $sample.process_working_set_mib = [math]::Round($Process.WorkingSet64 / 1MB, 1)
        $sample.process_private_mib = [math]::Round($Process.PrivateMemorySize64 / 1MB, 1)
    } catch {
        # The process may have exited between the status check and this sample.
    }

    if ($gpuCountersAvailable) {
        try {
            $adapters = @{}
            $counterSamples = Get-Counter -Counter $gpuCounterPaths -ErrorAction Stop | Select-Object -ExpandProperty CounterSamples
            foreach ($counter in $counterSamples) {
                $adapterId = $counter.InstanceName -replace "_part_\d+$", ""
                if (-not $adapters.ContainsKey($adapterId)) {
                    $adapters[$adapterId] = [ordered]@{
                        adapter = $adapterId
                        shared_memory_used_mib = $null
                        dedicated_memory_used_mib = $null
                        local_memory_used_mib = $null
                    }
                }

                $valueMiB = [math]::Round([double] $counter.CookedValue / 1MB, 1)
                if ($counter.Path -match "\\shared usage$") {
                    $adapters[$adapterId].shared_memory_used_mib = $valueMiB
                } elseif ($counter.Path -match "\\dedicated usage$") {
                    $adapters[$adapterId].dedicated_memory_used_mib = $valueMiB
                } elseif ($counter.Path -match "\\local usage$") {
                    $adapters[$adapterId].local_memory_used_mib = $valueMiB
                }
            }
            $sample.gpu = @($adapters.Values | ForEach-Object { [pscustomobject] $_ })
        } catch {
            Write-Warning "Could not query Windows GPU memory counters: $($_.Exception.Message)"
        }
    }

    [pscustomobject] $sample
}

function Get-LastTiming {
    param(
        [string] $Text,
        [ValidateSet("prompt", "generation")] [string] $Kind
    )

    $lines = $Text -split "`r?`n"
    if ($Kind -eq "prompt") {
        $line = $lines | Where-Object { $_ -match "prompt eval time" } | Select-Object -Last 1
    } else {
        $line = $lines | Where-Object { $_ -match "eval time" -and $_ -notmatch "prompt eval time" } | Select-Object -Last 1
    }
    if (-not $line) {
        return $null
    }

    $match = [regex]::Match($line, "(?<tokens>[0-9.]+) tokens per second")
    if ($match.Success) {
        return [double] $match.Groups["tokens"].Value
    }
    return $null
}

function Get-ResourceSummary {
    param([object[]] $Samples)

    if (-not $Samples -or $Samples.Count -eq 0) {
        return $null
    }

    $gpuSamples = @($Samples | ForEach-Object { $_.gpu } | Where-Object { $_ })
    [ordered]@{
        sample_count = $Samples.Count
        system_ram_used_mib = [ordered]@{
            max = [math]::Round(($Samples | Measure-Object system_ram_used_mib -Maximum).Maximum, 1)
            average = [math]::Round(($Samples | Measure-Object system_ram_used_mib -Average).Average, 1)
        }
        llama_process_working_set_mib = [ordered]@{
            max = [math]::Round(($Samples | Where-Object { $null -ne $_.process_working_set_mib } | Measure-Object process_working_set_mib -Maximum).Maximum, 1)
            average = [math]::Round(($Samples | Where-Object { $null -ne $_.process_working_set_mib } | Measure-Object process_working_set_mib -Average).Average, 1)
        }
        gpu_memory_used_mib_by_adapter = @(
            $gpuSamples | Group-Object adapter | ForEach-Object {
                [ordered]@{
                    adapter = $_.Name
                    shared_max = [math]::Round(($_.Group | Measure-Object shared_memory_used_mib -Maximum).Maximum, 1)
                    shared_average = [math]::Round(($_.Group | Measure-Object shared_memory_used_mib -Average).Average, 1)
                    dedicated_max = [math]::Round(($_.Group | Measure-Object dedicated_memory_used_mib -Maximum).Maximum, 1)
                    dedicated_average = [math]::Round(($_.Group | Measure-Object dedicated_memory_used_mib -Average).Average, 1)
                    local_max = [math]::Round(($_.Group | Measure-Object local_memory_used_mib -Maximum).Maximum, 1)
                    local_average = [math]::Round(($_.Group | Measure-Object local_memory_used_mib -Average).Average, 1)
                }
            }
        )
    }
}

$experiments = @(
    [pscustomobject]@{ Name = "baseline"; Settings = @{}; NeedsPredictLog = $false },
    [pscustomobject]@{ Name = "adapt"; Settings = @{ LLAMA_EXPERT_ADAPT = "1" }; NeedsPredictLog = $false },
    [pscustomobject]@{ Name = "predict"; Settings = @{ LLAMA_EXPERT_PREDICT = "1" }; NeedsPredictLog = $true },
    [pscustomobject]@{ Name = "rampool"; Settings = @{ LLAMA_EXPERT_RAMPOOL = $RamPoolGb.ToString([Globalization.CultureInfo]::InvariantCulture) }; NeedsPredictLog = $false },
    [pscustomobject]@{ Name = "pread"; Settings = @{ LLAMA_EXPERT_RAMPOOL = $RamPoolGb.ToString([Globalization.CultureInfo]::InvariantCulture); LLAMA_EXPERT_PREAD = "1" }; NeedsPredictLog = $false },
    [pscustomobject]@{ Name = "prefetch"; Settings = @{ LLAMA_EXPERT_PREFETCH_GB = "2"; LLAMA_EXPERT_PREFETCH_THREADS = "8" }; NeedsPredictLog = $false }
)

if ($Only.Count -gt 0) {
    $experiments = @($experiments | Where-Object { $Only -contains $_.Name })
    if ($experiments.Count -eq 0) {
        throw "No experiment matched -Only. Valid values: baseline, adapt, predict, rampool, pread, prefetch"
    }
}

$arguments = @(
    "-m", $Model,
    "-c", "2048",
    "-fit", "off",
    "-ngl", "0",
    "-p", "讲讲孙悟空大闹天宫的故事",
    "--temp", "0",
    "--reasoning-budget", "0",
    "-n", "200"
)

foreach ($experiment in $experiments) {
    $runId = "{0:yyyyMMdd-HHmmss}-{1}" -f (Get-Date), $experiment.Name
    $runDir = Join-Path $ResultsDir $runId
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $stdoutPath = Join-Path $runDir "llama.stdout.log"
    $stderrPath = Join-Path $runDir "llama.stderr.log"
    $statsPath = Join-Path $runDir "expert-stats.log"
    $predictLogPath = if ($experiment.NeedsPredictLog) { Join-Path $runDir "expert-predict.log" } else { $null }
    $samplesPath = Join-Path $runDir "resources.jsonl"
    $resultPath = Join-Path $runDir "result.json"

    Set-ExpertEnvironment -Settings $experiment.Settings -StatsPath $statsPath -PredictLogPath $predictLogPath
    $effectiveSettings = @{}
    Get-ChildItem Env:LLAMA_EXPERT_* | ForEach-Object { $effectiveSettings[$_.Name] = $_.Value }

    Write-Host "Running $($experiment.Name): $runDir"
    $startedAt = [DateTime]::UtcNow
    $process = Start-Process -FilePath $llamaCli -ArgumentList $arguments -WorkingDirectory $BinDir -NoNewWindow -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $samples = [System.Collections.Generic.List[object]]::new()

    do {
        $sample = Get-ResourceSample -Process $process
        $samples.Add($sample)
        ($sample | ConvertTo-Json -Compress -Depth 5) | Add-Content -LiteralPath $samplesPath -Encoding utf8
        Start-Sleep -Milliseconds $SampleIntervalMilliseconds
        $process.Refresh()
    } while (-not $process.HasExited)

    $sample = Get-ResourceSample -Process $process
    $samples.Add($sample)
    ($sample | ConvertTo-Json -Compress -Depth 5) | Add-Content -LiteralPath $samplesPath -Encoding utf8
    $finishedAt = [DateTime]::UtcNow

    $combinedLog = (Get-Content -LiteralPath $stdoutPath -Raw) + "`n" + (Get-Content -LiteralPath $stderrPath -Raw)
    $result = [ordered]@{
        experiment = $experiment.Name
        run_directory = $runDir
        started_utc = $startedAt.ToString("o")
        finished_utc = $finishedAt.ToString("o")
        duration_seconds = [math]::Round(($finishedAt - $startedAt).TotalSeconds, 3)
        exit_code = $process.ExitCode
        command = @($llamaCli) + $arguments
        environment = $effectiveSettings
        prompt_tokens_per_second = Get-LastTiming -Text $combinedLog -Kind prompt
        generation_tokens_per_second = Get-LastTiming -Text $combinedLog -Kind generation
        resource_summary = Get-ResourceSummary -Samples $samples.ToArray()
        files = [ordered]@{
            resource_samples = $samplesPath
            stdout = $stdoutPath
            stderr = $stderrPath
            expert_stats = $statsPath
            expert_predict = $predictLogPath
        }
    }
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resultPath -Encoding utf8

    if ($process.ExitCode -ne 0) {
        Write-Warning "Experiment $($experiment.Name) exited with code $($process.ExitCode). See $stderrPath"
    }
}

Write-Host "Completed. Results: $ResultsDir"
