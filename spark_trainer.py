import os
import json
import argparse
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
import torch
import torch.optim as optim
import torch.nn as nn
from model.cnn import JewelleryCNN

def train_one_batch(model, opt, crit, batch, device):
    # reconstruct tensor and reshape to (B,3,224,224)
    imgs = torch.stack([torch.tensor(d) for d in batch["data"]])
    imgs = imgs.view(-1, 3, 224, 224).to(device)
    lbls = torch.tensor(batch["labels"]).long().to(device)
    model.train()
    out = model(imgs)
    loss = crit(out, lbls)
    opt.zero_grad()
    loss.backward()
    opt.step()
    acc = (out.argmax(1) == lbls).float().mean().item()
    return loss.item(), acc

def train_partition(records_iter, num_classes):
    # initialize model & optimizer once per executor partition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JewelleryCNN(num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    for line in records_iter:
        batch = json.loads(line)
        loss, acc = train_one_batch(model, opt, crit, batch, device)
        print(f"[spark_trainer][partition] loss={loss:.4f}, acc={acc:.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Spark Streaming + PyTorch trainer for Jewellery dataset"
    )
    parser.add_argument(
        "--batch-folder",
        default="DS200-Lab04/batches/train",
        help="Folder streamed by stream_server.py (default: DS200-Lab04/batches/train)",
    )
    parser.add_argument(
        "--train-set",
        default="DS200-Lab04/data/Jewellery-Classification/dataset/training",
        help="Path to training dataset folders (bangles, bracelets, …)",
    )
    parser.add_argument(
        "--master",
        default="local[4]",
        help="Spark master URL (default: local[4])",
    )
    parser.add_argument(
        "--driver-memory",
        default="8g",
        help="Spark driver memory (default: 8g)",
    )
    parser.add_argument(
        "--executor-memory",
        default="8g",
        help="Spark executor memory (default: 8g)",
    )
    args = parser.parse_args()

    # infer number of classes
    classes = [
        d
        for d in os.listdir(args.train_set)
        if os.path.isdir(os.path.join(args.train_set, d))
    ]
    num_classes = len(classes)
    print(f"[spark_trainer] Detected {num_classes} classes: {classes}")

    # set Spark memory options
    os.environ[
        "PYSPARK_SUBMIT_ARGS"
    ] = f"--driver-memory {args.driver_memory} --executor-memory {args.executor_memory} pyspark-shell"

    conf = SparkConf().setAppName("JewelleryDL").setMaster(args.master)
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 2)

    # create a DStream of JSON lines
    stream = ssc.socketTextStream("localhost", 6100)

    # for each RDD, train per-partition
    stream.foreachRDD(
        lambda time, rdd: rdd.foreachPartition(lambda part: train_partition(part, num_classes))
    )

    print("[spark_trainer] Starting streaming context. Waiting for data...")
    ssc.start()
    ssc.awaitTermination()

if __name__ == "__main__":
    main()