$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }

Write-Host "Workspace root: $root"

$activate = Join-Path $root '.venv\Scripts\Activate.ps1'
if (Test-Path $activate) {
    Write-Host 'Activating venv...'
    & $activate
} else {
    Write-Host "No venv Activate.ps1 found at $activate — continuing with system Python"
}

$pdf = Join-Path $root 'new.pdf'
if (Test-Path $pdf) {
    Write-Host 'Found new.pdf — extracting images into static/data/'
    python "$root\extract_images_from_pdf.py" "$pdf"
} else {
    Write-Host 'No new.pdf found — skipping extraction'
}

New-Item -ItemType Directory -Path (Join-Path $root 'static\data') -Force | Out-Null

Write-Host 'Running image processing...'
python "$root\process_all_images.py"

if (Test-Path (Join-Path $root 'generate_report.py')) {
    Write-Host 'Generating CSV report...'
    python "$root\generate_report.py"
} else {
    Write-Host 'generate_report.py not found — skipping report generation'
}

$inputsZip = Join-Path $root 'all_inputs.zip'
$outputsZip = Join-Path $root 'all_outputs.zip'
Write-Host "Creating zip of all inputs -> $inputsZip"
Compress-Archive -Path (Join-Path $root 'static\data\*') -DestinationPath $inputsZip -Force
Write-Host "Creating zip of all processed outputs -> $outputsZip"
Compress-Archive -Path (Join-Path $root 'static\trained\*') -DestinationPath $outputsZip -Force

Write-Host "`nSummary"
Get-ChildItem -Path (Join-Path $root 'static\data') -File | Select-Object Name,LastWriteTime | Format-Table -AutoSize
Get-ChildItem -Path (Join-Path $root 'static\trained') -Recurse | Select-Object FullName,LastWriteTime | Format-Table -AutoSize
Write-Host "`nCreated zips:"
Get-Item $inputsZip, $outputsZip | Select-Object Name,FullName,Length,LastWriteTime | Format-Table -AutoSize

Write-Host 'Done.'
 $root = $PSScriptRoot
 if (-not $root) { $root = (Get-Location).Path }

 Write-Host "Workspace root: $root"

 $activate = Join-Path $root '.venv\Scripts\Activate.ps1'
 if (Test-Path $activate) {
     Write-Host 'Activating venv...'
     & $activate
 } else {
     Write-Host "No venv Activate.ps1 found at $activate — continuing with system Python"
 }

 $pdf = Join-Path $root 'new.pdf'
 if (Test-Path $pdf) {
     Write-Host 'Found new.pdf — extracting images into static/data/'
     python "$root\extract_images_from_pdf.py" "$pdf"
 } else {
     Write-Host 'No new.pdf found — skipping extraction'
 }

 New-Item -ItemType Directory -Path (Join-Path $root 'static\data') -Force | Out-Null

 Write-Host 'Running image processing...'
 python "$root\process_all_images.py"

 if (Test-Path (Join-Path $root 'generate_report.py')) {
     Write-Host 'Generating CSV report...'
     python "$root\generate_report.py"
 } else {
     Write-Host 'generate_report.py not found — skipping report generation'
 }

 $inputsZip = Join-Path $root 'all_inputs.zip'
 $outputsZip = Join-Path $root 'all_outputs.zip'
 Write-Host "Creating zip of all inputs -> $inputsZip"
 Compress-Archive -Path (Join-Path $root 'static\data\*') -DestinationPath $inputsZip -Force
 Write-Host "Creating zip of all processed outputs -> $outputsZip"
 Compress-Archive -Path (Join-Path $root 'static\trained\*') -DestinationPath $outputsZip -Force

 Write-Host "`nSummary"
 Get-ChildItem -Path (Join-Path $root 'static\data') -File | Select-Object Name,LastWriteTime | Format-Table -AutoSize
 Get-ChildItem -Path (Join-Path $root 'static\trained') -Recurse | Select-Object FullName,LastWriteTime | Format-Table -AutoSize
 Write-Host "`nCreated zips:"
 Get-Item $inputsZip, $outputsZip | Select-Object Name,FullName,Length,LastWriteTime | Format-Table -AutoSize

 Write-Host 'Done.'
$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }

Write-Host "Workspace root: $root"

$activate = Join-Path $root '.venv\Scripts\Activate.ps1'
if (Test-Path $activate) {
    Write-Host 'Activating venv...'
    & $activate
} else {
    Write-Host "No venv Activate.ps1 found at $activate — continuing with system Python"
}

$pdf = Join-Path $root 'new.pdf'
if (Test-Path $pdf) {
    Write-Host 'Found new.pdf — extracting images into static/data/'
    python "$root\extract_images_from_pdf.py" "$pdf"
} else {
    Write-Host 'No new.pdf found — skipping extraction'
}

New-Item -ItemType Directory -Path (Join-Path $root 'static\data') -Force | Out-Null

Write-Host 'Running image processing...'
python "$root\process_all_images.py"

if (Test-Path (Join-Path $root 'generate_report.py')) {
    Write-Host 'Generating CSV report...'
    python "$root\generate_report.py"
} else {
    Write-Host 'generate_report.py not found — skipping report generation'
}

$inputsZip = Join-Path $root 'all_inputs.zip'
$outputsZip = Join-Path $root 'all_outputs.zip'
Write-Host "Creating zip of all inputs -> $inputsZip"
Compress-Archive -Path (Join-Path $root 'static\data\*') -DestinationPath $inputsZip -Force
Write-Host "Creating zip of all processed outputs -> $outputsZip"
Compress-Archive -Path (Join-Path $root 'static\trained\*') -DestinationPath $outputsZip -Force

Write-Host "`nSummary"
Get-ChildItem -Path (Join-Path $root 'static\data') -File | Select-Object Name,LastWriteTime | Format-Table -AutoSize
Get-ChildItem -Path (Join-Path $root 'static\trained') -Recurse | Select-Object FullName,LastWriteTime | Format-Table -AutoSize
Write-Host "`nCreated zips:"
Get-Item $inputsZip, $outputsZip | Select-Object Name,FullName,Length,LastWriteTime | Format-Table -AutoSize

Write-Host 'Done.'
