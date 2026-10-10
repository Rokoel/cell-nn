# Install the official x64 runtime before TensorFlow/Qt are imported.
$ErrorActionPreference = 'Stop'
$installer = Join-Path $env:TEMP 'fish-vc_redist.x64.exe'
try {
    Invoke-WebRequest 'https://aka.ms/vc14/vc_redist.x64.exe' -OutFile $installer
    $signature = Get-AuthenticodeSignature $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
        throw 'The Visual C++ installer does not have a valid Microsoft signature.'
    }
    $process = Start-Process -FilePath $installer -ArgumentList '/install', '/passive', '/norestart' -Wait -PassThru
    if ($process.ExitCode -eq 3010) {
        Write-Host 'Visual C++ runtime installed. Windows requires a restart before analysis.'
    } elseif ($process.ExitCode -notin @(0,1638)) {
        throw "Visual C++ runtime installation failed (exit $($process.ExitCode))."
    }
} finally {
    Remove-Item $installer -ErrorAction SilentlyContinue
}
