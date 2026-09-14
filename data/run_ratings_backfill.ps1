Set-Location C:\Users\williamH\OpenFPL
$env:PYTHONIOENCODING = "utf-8"
$log = "C:\Users\williamH\OpenFPL\data\bbc_ratings_backfill.log"
foreach ($s in @("2025-26", "2024-25", "2026-27")) {
  python -m acquire backfill --source bbc_ratings --season $s 2>&1 | Out-File -Append -Encoding utf8 $log
}
"RATINGS DONE" | Out-File -Append -Encoding utf8 $log
