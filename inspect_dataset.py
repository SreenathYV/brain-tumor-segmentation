import pyarrow.parquet as pq
from pathlib import Path

folder = Path(r"C:\Users\yvsre\Desktop\CV_Project\Dataset")

for path in sorted(folder.glob("*.parquet")):
    parquet = pq.ParquetFile(path)
    print(f"\n{path.name}: {parquet.metadata.num_rows:,} rows")
    print(parquet.schema)

    row = next(parquet.iter_batches(batch_size=1)).to_pylist()[0]
    print("Sample fields:", {
        key: (type(value).__name__, list(value)[:5] if isinstance(value, dict) else str(value)[:120])
        for key, value in row.items()
    })