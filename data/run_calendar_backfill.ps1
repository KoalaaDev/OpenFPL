Set-Location C:\Users\williamH\OpenFPL
$env:PYTHONIOENCODING = "utf-8"
$log = "C:\Users\williamH\OpenFPL\data\bbc_calendar_backfill.log"
foreach ($s in @("2023-24", "2024-25", "2025-26", "2026-27")) {
  python -m acquire backfill --source bbc_calendar --season $s 2>&1 | Out-File -Append -Encoding utf8 $log
}
"CALENDAR DONE" | Out-File -Append -Encoding utf8 $log
