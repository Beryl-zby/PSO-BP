$ErrorActionPreference = 'Stop'
$fontTarget = Join-Path $PSScriptRoot '..\public\plot-fonts'
New-Item -ItemType Directory -Path $fontTarget -Force | Out-Null
foreach ($fontName in @('simsun.ttc', 'times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf')) {
    $fontSource = Join-Path $env:WINDIR "Fonts\$fontName"
    if (Test-Path -LiteralPath $fontSource) {
        Copy-Item -LiteralPath $fontSource -Destination (Join-Path $fontTarget $fontName) -Force
    } else { Write-Warning "System font missing: $fontName" }
}
Write-Output 'Local plot fonts configured.'
