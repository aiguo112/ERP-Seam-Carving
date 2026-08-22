$src = "D:\ERP_Seam_Carving\TemplateAMICT2026.doc"
$out = "D:\ERP_Seam_Carving\paper\amict_extract.txt"
$tout = "D:\ERP_Seam_Carving\paper\amict_tables.txt"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $word.Documents.Open($src, $false, $true)
$sb = New-Object System.Text.StringBuilder
$ps = $doc.PageSetup
[void]$sb.AppendLine("PARAS=$($doc.Paragraphs.Count) TABLES=$($doc.Tables.Count)")
[void]$sb.AppendLine("MARGINS_PT L=$($ps.LeftMargin) R=$($ps.RightMargin) T=$($ps.TopMargin) B=$($ps.BottomMargin) GUTTER=$($ps.Gutter)")
[void]$sb.AppendLine("PAGE W=$($ps.PageWidth) H=$($ps.PageHeight) ORIENT=$($ps.Orientation) COLS=$($ps.TextColumns.Count)")
if ($ps.TextColumns.Count -gt 0) {
  [void]$sb.AppendLine("COL_WIDTH=$($ps.TextColumns.Item(1).Width) COL_SPACING=$($ps.TextColumns.Spacing)")
}
[void]$sb.AppendLine("---PARAS---")
$n = [Math]::Min($doc.Paragraphs.Count, 180)
for ($i = 1; $i -le $n; $i++) {
  $p = $doc.Paragraphs.Item($i)
  $rng = $p.Range
  $snip = ($rng.Text -replace "[\r\x07]", " ").Trim()
  if ($snip.Length -gt 240) { $snip = $snip.Substring(0, 240) }
  if ($snip.Length -gt 0) {
    $font = $rng.Font
    [void]$sb.AppendLine("[$i] font=$($font.Name) size=$($font.Size) bold=$($font.Bold) italic=$($font.Italic) align=$($p.Alignment) spacebefore=$($p.SpaceBefore) spaceafter=$($p.SpaceAfter) | $snip")
  }
}
[void]$sb.AppendLine("")
[void]$sb.AppendLine("=====FULL TEXT=====")
[void]$sb.AppendLine(($doc.Content.Text -replace "`r", "`n"))
[System.IO.File]::WriteAllText($out, $sb.ToString(), [System.Text.Encoding]::UTF8)

$tb = New-Object System.Text.StringBuilder
for ($t = 1; $t -le $doc.Tables.Count; $t++) {
  $table = $doc.Tables.Item($t)
  [void]$tb.AppendLine("==== TABLE $t rows=$($table.Rows.Count) cols=$($table.Columns.Count) ====")
  $nr = [Math]::Min($table.Rows.Count, 20)
  $nc = [Math]::Min($table.Columns.Count, 8)
  for ($r = 1; $r -le $nr; $r++) {
    $cells = @()
    for ($c = 1; $c -le $nc; $c++) {
      try {
        $txt = ($table.Cell($r, $c).Range.Text -replace "[\r\x07]", " ").Trim()
        $cells += $txt
      } catch {}
    }
    [void]$tb.AppendLine(($cells -join " | "))
  }
}
[System.IO.File]::WriteAllText($tout, $tb.ToString(), [System.Text.Encoding]::UTF8)
$doc.Close($false)
$word.Quit()
Write-Output "wrote $out and $tout"
