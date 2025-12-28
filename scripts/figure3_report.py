def cal_unsupported_reduction(data):
    for dataset, values in data.items():
        reduction_ratio = (values["WGA"][0] - values["WGA+ChimeraLM"][0]) / values[
            "WGA"
        ][0]
        print(f"{dataset} {reduction_ratio}")


def main():
    data_compared_with_gold = {
        "p2": {"WGA": [3601099, 8815], "WGA+ChimeraLM": [297503, 8067]},
        "mk1c": {"WGA": [444197, 7193], "WGA+ChimeraLM": [32163, 6269]},
    }

    data_compared_with_bulk = {
        "p2": {"WGA": [3596122, 13792], "WGA+ChimeraLM": [295656, 9914]},
        "mk1c": {"WGA": [445743, 5647], "WGA+ChimeraLM": [33515, 4917]},
    }

    print("Compare with long and short read data")
    cal_unsupported_reduction(data_compared_with_gold)
    print("Compare with only long read data")
    cal_unsupported_reduction(data_compared_with_bulk)


if __name__ == "__main__":
    main()
