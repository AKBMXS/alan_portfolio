#!wget -O Stock_news.csv \
# https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv
#!/usr/bin/env python3
import os, sys, csv, argparse, logging, pandas as pd

csv.field_size_limit(sys.maxsize)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(levelname)s: %(message)s")
logger = logging.getLogger()

def parse_args():
    p = argparse.ArgumentParser(
        description="Shard Stock_news.csv into ticker-filtered Parquet files"
    )
    p.add_argument("--csv",      required=True,
                   help="Path to Stock_news.csv")
    p.add_argument("--last_row", type=int, default=0,
                   help="Rows to skip (header=0)")
    p.add_argument("--batch_size",type=int, default=2000,
                   help="Rows per Parquet shard")
    p.add_argument("--tickers",  nargs="+", required=True,
                   help="Tickers to filter")
    p.add_argument("--output_dir",required=True,
                   help="Directory for Parquet shards")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine next shard index
    existing = sorted(f for f in os.listdir(args.output_dir)
                      if f.startswith("fnspid_") and f.endswith(".parquet"))
    shard_idx = len(existing)
    logger.info(f"Resuming from shard {shard_idx:04d}")

    # Read header to get fieldnames
    with open(args.csv, newline="", encoding="utf8") as f:
        header = f.readline().rstrip("\n")
    fieldnames = next(csv.reader([header]))
    # Open trimmed CSV (header removed, skip rows)
    f_rem = open(args.csv, newline="", encoding="utf8")
    reader = csv.DictReader(f_rem, fieldnames=fieldnames)
    # Skip header row plus any last_row
    next(reader)
    for _ in range(args.last_row):
        next(reader, None)

    buffer = []
    filtered = scanned = 0

    for rec in reader:
        scanned += 1
        sym = rec.get("Stock_symbol")
        if not sym or sym not in args.tickers:
            continue
        filtered += 1
        buffer.append({
            "symbol":    sym,
            "date":      rec.get("Date",""),
            "title":     rec.get("Article_title",""),
            "url":       rec.get("Url",""),
            "publisher": rec.get("Publisher",""),
            "author":    rec.get("Author",""),
            "body":      rec.get("Article","")
        })
        if len(buffer) >= args.batch_size:
            df = pd.DataFrame(buffer).drop_duplicates(["symbol","date","title"])
            path = os.path.join(args.output_dir, f"fnspid_{shard_idx:04d}.parquet")
            df.to_parquet(path, index=False, compression="snappy")
            logger.info(f"Wrote shard {shard_idx:04d}: {len(df)} rows → {path}")
            buffer.clear(); shard_idx += 1

    # Final flush
    if buffer:
        df = pd.DataFrame(buffer).drop_duplicates(["symbol","date","title"])
        path = os.path.join(args.output_dir, f"fnspid_{shard_idx:04d}.parquet")
        df.to_parquet(path, index=False, compression="snappy")
        logger.info(f"Wrote final shard {shard_idx:04d}: {len(df)} rows → {path}")

    logger.info(f"✅ Done. Scanned={scanned}, Filtered={filtered}")

if __name__=="__main__":
    main()

#run:
# !python3 shard_fnspid_csv_stream.py \
#   --csv Stock_news.csv \
#   --last_row 0 \
#   --batch_size 2000 \
#   --tickers AAPL ABBV ABT ACN ADBE AIG AMD AMGN AMT AMZN \
#             AVGO AXP BA BAC BK BKNG BLK BMY BRK.B C CAT CHTR CL \
#             CMCSA COF COP COST CRM CSCO CVS CVX DE DHR DIS DUK \
#             EMR FDX GD GE GILD GM GOOG GOOGL GS HD HON IBM INTU \
#             ISRG JNJ JPM KO LIN LLY LMT LOW MA MCD MDLZ MDT MET \
#             META MMM MO MRK MS MSFT NEE NFLX NKE NOW NVDA OXY PEP \
#             PFE PG PLTR PM PYPL QCOM RTX SBUX SCHW SO SPG T TGT \
#             TMO TMUS TSLA TXN UNH UNP UPS USB V VZ WFC WMT XOM \
#             RELIANCE.NS TCS.NS HDFCBANK.NS ICICIBANK.NS \
#             HINDUNILVR.NS INFY.NS ITC.NS BHARTIARTL.NS SBIN.NS \
#             BAJFINANCE.NS ABB.NS ACC.NS APLAPOLLO.NS DABUR.NS \
#             CUMMINSIND.NS \
#   --output_dir fnspid_shards_full