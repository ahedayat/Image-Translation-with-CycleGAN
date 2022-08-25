import os
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


def preprocess_df(df, train_val_ratio=0.8):
    """
        Preprocessing Dataframe 
    """
    df["domain_name"] = df["domain"].apply(lambda x: x[3:-1])
    df["domain"] = df["domain"].apply(lambda x: x[0])

    df_test = df[df["split"] == "test"]
    df_train = df[df["split"] == "train"]

    df_train_a = df_train[df_train["domain"] == "A"]
    df_train_b = df_train[df_train["domain"] == "B"]

    df_val_a = df_train_a.sample(frac=1 - train_val_ratio)
    df_val_b = df_train_b.sample(frac=1 - train_val_ratio)

    df_val = pd.concat([df_val_a, df_val_b])
    df_val = df_val.sample(frac=1)
    df_val["split"] = "val"

    df_train = df_train.drop(df_val.index)

    df = pd.concat([df_train, df_val, df_test])

    return df


def add_train(domain, train_val_ratio):
    """
        Adding train data of specific domain that is exist in 'train{$DOMAIN}' to
        data frame and split them to 'train' and 'val' data.
    """
    assert domain in ["A", "B"], "domain must be one of this items: ['A', 'B']"

    df_train = pd.DataFrame(
        columns=["image_id", "domain", "split", "image_path", "domain_name"])

    train_dir = "train" + domain
    train_files = os.listdir(os.path.join(".", train_dir))

    num_train = math.floor(train_val_ratio * len(train_files))
    num_val = len(train_files) - num_train

    splits = ["train"] * num_train + ["val"] * num_val

    random.shuffle(splits)

    with tqdm(train_files) as t_files:
        for ix, (image_name) in enumerate(t_files):
            t_files.set_description(f"Adding Train{domain} Data")

            image_id = image_name.split(".")[0]
            split = "train"
            image_path = os.path.join(train_dir, image_name)
            domain_name = "Horse" if domain == "A" else "Zebra"

            df_train = df_train.append({
                "image_id": image_id,
                "domain": domain,
                "split": splits[ix],
                "image_path": image_path,
                "domain_name": domain_name,
            }, ignore_index=True)

    # Split data frame to

    return df_train


def add_test(domain):
    """
        Adding test data of specific domain that is exist in 'test{$DOMAIN}' to
        data frame.
    """
    assert domain in ["A", "B"], "domain must be one of this items: ['A', 'B']"

    df_test = pd.DataFrame(
        columns=["image_id", "domain", "split", "image_path", "domain_name"])

    test_dir = "test" + domain
    test_files = os.listdir(os.path.join(".", test_dir))

    with tqdm(test_files) as t_files:
        for image_name in t_files:
            t_files.set_description(f"Adding Test{domain} Data")

            image_id = image_name.split(".")[0]
            split = "test"
            image_path = os.path.join(test_dir, image_name)
            domain_name = "Horse" if domain == "A" else "Zebra"

            df_test = df_test.append({
                "image_id": image_id,
                "domain": domain,
                "split": split,
                "image_path": image_path,
                "domain_name": domain_name,
            }, ignore_index=True)

    return df_test


def _main():
    # Reading Pathdf
    csv_path = "./metadata.csv"

    # Train and Validation split ratio
    train_val_ratio = 0.8

    # Saving Path
    saving_path = "./preprocessed.csv"

    # Reading Dataframe
    df = pd.read_csv(csv_path)

    # Preprocessing Dataframe
    df = preprocess_df(df, train_val_ratio=train_val_ratio)

    # Ignoring Test Data of 'metadata.csv'
    df = df[df["split"] != "test"]

    # Adding TrainA to the Dataframe
    df_train_a = add_train(domain="A", train_val_ratio=train_val_ratio)
    # Adding TrainB to the Dataframe
    df_train_b = add_train(domain="B", train_val_ratio=train_val_ratio)

    # Adding TestA to the Dataframe
    df_test_a = add_test(domain="A")
    # Adding TestB to the Dataframe
    df_test_b = add_test(domain="B")

    df = pd.concat([df_train_a, df_train_b, df_test_a, df_test_b])

    # Saving Dataframe
    df.to_csv(saving_path)


if __name__ == "__main__":
    _main()
