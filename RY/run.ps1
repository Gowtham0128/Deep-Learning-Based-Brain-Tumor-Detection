param()
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (Test-Path "$root\.venv\Scripts\Activate.ps1") {
    & "$root\.venv\Scripts\Activate.ps1"
}
python "$root\process_all_images.py"