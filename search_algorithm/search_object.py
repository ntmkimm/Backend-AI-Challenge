import polars as pl
import re

ROOT = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1/"
df = pl.read_parquet(ROOT + "objects.parquet")

command = "person=8 car>2"

# Tách từng điều kiện
object_filters = command.strip().split()

# Parse từng điều kiện và xây mảng conditions
conditions = []
for expr in object_filters:
    match = re.match(r"(\w+)([<>=]+)(\d+)", expr)
    if not match:
        raise ValueError(f"Invalid filter expression: {expr}")

    col_name, op, value = match.groups()
    col = pl.col(col_name)
    value = int(value)

    if op == "=":
        conditions.append(col == value)
    elif op == ">":
        conditions.append(col > value)
    elif op == ">=":
        conditions.append(col >= value)
    elif op == "<":
        conditions.append(col < value)
    elif op == "<=":
        conditions.append(col <= value)
    else:
        raise ValueError(f"Unsupported operator: {op}")

# Kết hợp các điều kiện bằng AND
if not conditions:
    raise ValueError("No valid filters provided.")

combined_condition = conditions[0]
for cond in conditions[1:]:
    combined_condition &= cond

# Lọc dataframe
result = df.filter(combined_condition)

# Hiển thị kết quả
# print(result.select(["filepath"]))
for row in result.iter_rows(named=True):
    print(row["filepath"])
    print(row["frame_id"])
# print(result.select(["filepath"]).row(0)[0])

