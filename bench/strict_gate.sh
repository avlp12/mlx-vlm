#!/bin/sh
# Stricter than the shared advisory gate: require BOTH no foreign heavy python
# AND GPU device utilization below 25%, twice in a row 10s apart.
ok() {
  b=$(ps -Ao pid,rss,args | awk '$2>8388608 && tolower($0) ~ /python/ {print $1}')
  [ -n "$b" ] && return 1
  u=$(ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "$u" ] && [ "$u" -gt 25 ] && return 1
  return 0
}
i=0
while [ $i -lt 90 ]; do
  if ok; then sleep 10; if ok; then echo "QUIET"; exit 0; fi; fi
  sleep 20; i=$((i+1))
done
echo "TIMEOUT"; exit 1
