# Verify nbytes vs dims per dtype code (field 2) using the hypothesised layout formulas.
# Usage: awk -F'\t' -f verify.awk table.txt
NR == 1 { next }
{
  split(substr($4, 2, length($4) - 2), d, ",")
  # numel = product of dims (dims beyond rank are 0 -> treat 0 as absent, but rank comes from kind)
  numel = 1; nd = 0
  for (i = 1; i <= 4; i++) if (d[i] > 0) { numel *= d[i]; nd++ }
  e = -1; ok = "?"
  if ($2 == 24) { k16 = int((13 * d[2]) / 16); e = d[1] * int((k16 + 15) / 16) * 16 }  # q6g64: N * pad16(13K/16)
  else if ($2 == 7) { if (numel % 16 == 0) e = numel * 17 / 16 }      # q8g64: numel*17/16
  else if ($2 == 5) { k9 = int((9 * d[nd]) / 16); e = 64 + int(numel / d[nd]) * int((k9 + 15) / 16) * 16 }  # q4c: 64 + (numel/K) * pad16(9K/16)
  else if ($2 == 16) { if (numel % 2 == 0) e = numel / 2 }            # ht: numel/2 (4-bit)
  else if ($2 == 23) { if (numel % 64 == 0) e = numel * 33 / 64 }     # i4r: 33 B per 64 elems
  else if ($2 == 0 || $2 == 2) { e = numel * 2 }                       # bf16/f16: 2 B/elem
  else if ($2 == 4) { e = numel * 8 }                                  # i64: 8 B/elem
  if (e >= 0) ok = (e == $6) ? "OK" : "MISMATCH"
  cnt[$2]++
  if (ok != "OK") { bad[$2]++; print "MISMATCH", $2, $1, $4, $6, "expected", e }
}
END {
  for (k in cnt) printf "dtype %s: %d tensors, %d mismatches\n", k, cnt[k], bad[k] + 0
}
