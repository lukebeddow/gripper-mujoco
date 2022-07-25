#!/usr/bin/env python3

# 0, 4, 8, 12 are ends
the_readings = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

n_readings = 5
prev_steps = 3

total_readings = (n_readings - 1) * prev_steps
print("total readings is", total_readings)

result = [-1 for i in range(2 * prev_steps + 1)]
print("length of result is", len(result))

result[0] = the_readings[total_readings]

for i in range(prev_steps):

  first_sample = total_readings - i * (n_readings - 1)
  print("first sample is", first_sample, "which is", the_readings[first_sample])

  # result[i * 2] = the_readings[first_sample]
  result[i * 2 + 2] = the_readings[first_sample - (n_readings - 1)]
  result[i * 2 + 1] = result[i * 2 + 2] - result[i * 2]
  print("result is", result)

print("\nfinal result is", result)