Set-Location C:\Users\williamH\OpenFPL
$env:PYTHONIOENCODING = "utf-8"
$log = "C:\Users\williamH\OpenFPL\data\bbc_history_backfill.log"
python -m acquire backfill --source bbc_calendar --season 2022-23 2>&1 | Out-File -Append -Encoding utf8 $log
foreach ($s in @("2023-24", "2022-23")) {
  python -m acquire backfill --source bbc --season $s 2>&1 | Out-File -Append -Encoding utf8 $log
}
"HISTORY DONE" | Out-File -Append -Encoding utf8 $log
