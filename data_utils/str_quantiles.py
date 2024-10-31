import math


def str_quantiles(str_min: str, str_max: str, n: int):
  if str_max < str_min:
    str_min, str_max = str_max, str_min
  if len(str_min) == 0 and len(str_max) == 0:
    return [""]
  ord_min = ord(str_min[0]) if str_min else 0
  ord_max = ord(str_max[0]) + 1
  if (ord_max - ord_min) >= n:
    return [
      chr(ord_q) if ord_q else ""
      for ord_q in int_quantiles(ord_min, ord_max, n)
    ]
  sub_n = math.ceil(n / (ord_max - ord_min))
  return [
    f"{chr(ord_this)}{sub_char}"
    for sub_char in str_quantiles(str_min[1:], str_max[1:], sub_n)
    for ord_this in range(ord_min, ord_max + 1)
  ]


def int_quantiles(min: int, max: int, n: int):
  if min == max:
    return [min]
  if n == 1:
    return [min, max]
  return [min + i * (max - min) // n for i in range(n + 1)]
