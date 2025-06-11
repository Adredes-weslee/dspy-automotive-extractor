# Create data directory if it doesn't exist
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
}

Write-Host "=== CHECKING AVAILABLE NHTSA DATA (2010-2025) ==="

# Test different years and months to see what's available
$availableFiles = @()
$years = @(2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)

foreach ($year in $years) {
    Write-Host "Checking year $year..."
    
    for ($month = 1; $month -le 12; $month++) {
        $monthStr = $month.ToString("00")
        $url = "https://static.nhtsa.gov/odi/ffdd/sgo-$year-$monthStr/SGO-$year-$monthStr" + "_Incident_Reports_ADAS.csv"
        $filename = "data/NHTSA_complaints_$year`_$monthStr.csv"
        
        # Test if URL exists with a quick HEAD request
        try {
            $response = Invoke-WebRequest -Uri $url -Method Head -ErrorAction Stop -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Host "✅ Found: $year-$monthStr" -ForegroundColor Green
                $availableFiles += @{
                    Year = $year
                    Month = $monthStr
                    URL = $url
                    Filename = $filename
                }
            }
        }
        catch {
            # Only show missing for recent years to reduce noise
            if ($year -ge 2020) {
                Write-Host "❌ Missing: $year-$monthStr" -ForegroundColor DarkGray
            }
        }
    }
}

Write-Host ""
Write-Host "=== DOWNLOADING AVAILABLE FILES ==="
Write-Host "Found $($availableFiles.Count) available files across all years"

if ($availableFiles.Count -eq 0) {
    Write-Host "No files available to download!"
    exit
}

# Show what we found
Write-Host "Available data periods:"
foreach ($file in $availableFiles) {
    Write-Host "  - $($file.Year)-$($file.Month)"
}
Write-Host ""

foreach ($file in $availableFiles) {
    Write-Host "Downloading $($file.Year)-$($file.Month)..."
    curl.exe -L -k $file.URL -o $file.Filename
    
    if (Test-Path $file.Filename) {
        # Check if file contains actual CSV data (not an error XML)
        $firstLine = Get-Content $file.Filename -First 1
        if ($firstLine -like "*<?xml*" -or $firstLine -like "*<Error>*") {
            Write-Host "❌ Error response (removing invalid file): $($file.Filename)" -ForegroundColor Red
            Remove-Item $file.Filename
        } else {
            $size = [math]::Round((Get-Item $file.Filename).Length / 1KB, 1)
            Write-Host "✅ Downloaded: $($file.Filename) ($size KB)"
        }
    } else {
        Write-Host "❌ Failed: $($file.Filename)"
    }
}

# Combine all valid downloaded files
Write-Host ""
Write-Host "=== COMBINING ALL AVAILABLE DATA ==="

$csvFiles = Get-ChildItem -Path "data" -Filter "NHTSA_complaints_*.csv" | Sort-Object Name
$outputFile = "data/NHTSA_complaints_all.csv"

if ($csvFiles.Count -gt 0) {
    Write-Host "Found $($csvFiles.Count) valid CSV files to combine..."
    
    # Copy first file with headers
    Copy-Item $csvFiles[0].FullName $outputFile
    Write-Host "Base file: $($csvFiles[0].Name)"
    
    # Append remaining files without headers
    for ($i = 1; $i -lt $csvFiles.Count; $i++) {
        $content = Get-Content $csvFiles[$i].FullName | Select-Object -Skip 1
        Add-Content -Path $outputFile -Value $content
        Write-Host "Added: $($csvFiles[$i].Name)"
    }
    
    # Also create the standard filename for compatibility
    Copy-Item $outputFile "data/NHTSA_complaints.csv"
    
    $totalRows = (Get-Content $outputFile | Measure-Object -Line).Lines - 1
    $fileSize = [math]::Round((Get-Item $outputFile).Length / 1MB, 2)
    
    Write-Host ""
    Write-Host "=== FINAL RESULT ==="
    Write-Host "Combined file: $outputFile"
    Write-Host "Also saved as: data/NHTSA_complaints.csv"
    Write-Host "Total rows: $totalRows"
    Write-Host "File size: $fileSize MB"
    Write-Host "Available periods: $($csvFiles.Count) months/years"
    
    # Show year range
    $years = $csvFiles | ForEach-Object { $_.Name -replace "NHTSA_complaints_(\d{4})_\d{2}\.csv", '$1' } | Sort-Object -Unique
    Write-Host "Data spans: $($years -join ', ')"
    
    # Optional: Clean up individual files
    $cleanup = Read-Host "Delete individual monthly files? (y/N)"
    if ($cleanup -eq "y" -or $cleanup -eq "Y") {
        $csvFiles | Remove-Item
        Write-Host "Cleaned up individual files"
    }
    
} else {
    Write-Host "No valid CSV files found!"
}

Write-Host ""
Write-Host "=== DOWNLOAD COMPLETE ==="
Write-Host "Your expanded dataset is ready for DSPy training!"