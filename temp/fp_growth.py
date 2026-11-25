
# 1. SỬ DỤNG THƯ VIỆN (MLXTEND) CHO THUẬT TOÁN FP-GROWTH


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

RANDOM_STATE = 42


def run_fpgrowth_library(transactions, min_support=0.20):
    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    one_hot_df = pd.DataFrame(encoded_array, columns=encoder.columns_)

    library_result = fpgrowth(
        one_hot_df,
        min_support=min_support,
        use_colnames=True
    )

    library_patterns = {
        frozenset(itemset): float(support)
        for itemset, support in zip(
            library_result["itemsets"],
            library_result["support"]
        )
    }
    return library_patterns, one_hot_df



# 2. TỰ CÀI ĐẶT THUẬT TOÁN FP-GROWTH (KHÔNG DÙNG CLASS, DÙNG DICT & HÀM THƯỜNG)


def create_fp_node(item, count, parent):
    # Tạo một nút trong cây FP-Tree bằng dictionary
    return {
        "item": item,
        "count": count,
        "parent": parent,
        "children": {},
        "node_link": None,
    }


def create_header_entry(support):
    # Tạo một mục trong bảng Header bằng dictionary
    return {
        "support": support,
        "head": None,
        "tail": None,
    }


def header_add_node(header_entry, node):
    # Nối thêm nút mới vào danh sách liên kết của cùng item trong header table
    if header_entry["head"] is None:
        header_entry["head"] = node
    else:
        header_entry["tail"]["node_link"] = node
    header_entry["tail"] = node


def build_fp_tree(weighted_transactions, min_support_count):
    # Đếm support của từng item từ các giao dịch có trọng số.
    item_support = {}
    for transaction, weight in weighted_transactions:
        for item in set(transaction):
            item_support[item] = item_support.get(item, 0) + weight

    # Loại item không đạt minimum support.
    frequent_items = {
        item: count
        for item, count in item_support.items()
        if count >= min_support_count
    }

    if not frequent_items:
        return None, {}

    header = {
        item: create_header_entry(count)
        for item, count in frequent_items.items()
    }

    root = create_fp_node(None, 0, None)

    # Sắp xếp item theo support giảm dần để tạo các prefix path dùng chung.
    for transaction, weight in weighted_transactions:
        filtered = [item for item in transaction if item in frequent_items]
        filtered.sort(key=lambda item: (-frequent_items[item], str(item)))

        current = root
        for item in filtered:
            if item in current["children"]:
                child = current["children"][item]
                child["count"] += weight
            else:
                child = create_fp_node(item, weight, current)
                current["children"][item] = child
                header_add_node(header[item], child)
            current = child

    return root, header


def conditional_pattern_base(header_entry):
    # Truy ngược từ từng node của item về root để lấy prefix path.
    pattern_base = []
    node = header_entry["head"]

    while node is not None:
        path = []
        parent = node["parent"]

        while parent is not None and parent["item"] is not None:
            path.append(parent["item"])
            parent = parent["parent"]

        if path:
            pattern_base.append((path, node["count"]))

        node = node["node_link"]

    return pattern_base


def mine_fp_tree(header, min_support_count, prefix, frequent_patterns):
    # Khai phá từ item có support thấp lên cao để sinh các frequent itemsets.
    ordered_items = sorted(
        header.keys(),
        key=lambda item: (header[item]["support"], str(item))
    )

    for item in ordered_items:
        new_pattern = prefix | frozenset([item])
        frequent_patterns[new_pattern] = header[item]["support"]

        # Xây dựng conditional pattern base và conditional FP-tree.
        conditional_base = conditional_pattern_base(header[item])
        conditional_tree, conditional_header = build_fp_tree(
            conditional_base,
            min_support_count
        )

        if conditional_tree is not None:
            mine_fp_tree(
                conditional_header,
                min_support_count,
                new_pattern,
                frequent_patterns
            )


def fp_growth(transactions, min_support=0.2):
    # Hàm chính: chuyển dữ liệu thành giao dịch có trọng số, dựng FP-tree và khai phá.
    if not transactions:
        return {}

    min_support_count = int(np.ceil(min_support * len(transactions)))
    weighted_transactions = [(list(set(t)), 1) for t in transactions]

    tree, header = build_fp_tree(weighted_transactions, min_support_count)

    frequent_patterns = {}
    if tree is not None:
        mine_fp_tree(header, min_support_count, frozenset(), frequent_patterns)

    # Chuyển support count thành support ratio để dễ so sánh với mlxtend.
    return {
        itemset: count / len(transactions)
        for itemset, count in frequent_patterns.items()
    }


def format_itemsets(patterns):
    # Chuyển itemset dạng frozenset thành chuỗi để hiển thị trong DataFrame.
    rows = [
        {"itemsets": tuple(sorted(itemset)), "support": support}
        for itemset, support in patterns.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["itemsets", "support"])
    return pd.DataFrame(rows).sort_values(
        ["support", "itemsets"],
        ascending=[False, True]
    ).reset_index(drop=True)



# 3. DATA MOCK VÀ KIỂM THỬ HAI THUẬT TOÁN


if __name__ == "__main__":
    transactions = [
        ["Bread", "Milk"],
        ["Bread", "Diaper", "Beer", "Eggs"],
        ["Milk", "Diaper", "Beer", "Cola"],
        ["Bread", "Milk", "Diaper", "Beer"],
        ["Bread", "Milk", "Diaper", "Cola"],
        ["Bread", "Milk", "Eggs"],
        ["Milk", "Diaper", "Beer"],
        ["Bread", "Diaper", "Cola"],
        ["Bread", "Milk", "Diaper", "Beer", "Eggs"],
        ["Milk", "Beer", "Cola"],
        ["Bread", "Milk", "Diaper"],
        ["Bread", "Beer", "Eggs"],
        ["Milk", "Diaper", "Cola"],
        ["Bread", "Milk", "Beer"],
        ["Bread", "Diaper", "Beer", "Cola"],
        ["Milk", "Eggs"],
        ["Bread", "Milk", "Diaper", "Beer"],
        ["Bread", "Cola"],
        ["Milk", "Diaper", "Beer", "Eggs"],
        ["Bread", "Milk", "Diaper", "Cola"]
    ]

    MIN_SUPPORT = 0.20
    print("Số giao dịch:", len(transactions))
    print("Minimum support:", MIN_SUPPORT)
    print("Minimum support count:", int(np.ceil(MIN_SUPPORT * len(transactions))))

    # Chạy bằng thư viện mlxtend
    library_patterns, one_hot_df = run_fpgrowth_library(transactions, min_support=MIN_SUPPORT)
    print("Số frequent itemsets từ mlxtend:", len(library_patterns))

    # Chạy FP-Growth tự cài đặt
    custom_patterns = fp_growth(transactions, min_support=MIN_SUPPORT)
    print("Số frequent itemsets tự cài đặt:", len(custom_patterns))
    print(format_itemsets(custom_patterns).to_string(index=False))

    # So sánh hai kết quả theo itemset và support
    all_itemsets = set(library_patterns) | set(custom_patterns)
    comparison_rows = []
    for itemset in all_itemsets:
        lib_support = library_patterns.get(itemset)
        custom_support = custom_patterns.get(itemset)
        comparison_rows.append({
            "Itemset": tuple(sorted(itemset)),
            "Library_Support": lib_support,
            "Custom_Support": custom_support,
            "Match": (
                lib_support is not None
                and custom_support is not None
                and np.isclose(lib_support, custom_support, atol=1e-12)
            )
        })

    comparison_df = pd.DataFrame(comparison_rows).sort_values("Itemset").reset_index(drop=True)
    print("\nTổng itemset:", len(all_itemsets))
    print("Itemset trùng khớp:", comparison_df["Match"].sum())
    print("Hai kết quả giống nhau:", len(library_patterns) == len(custom_patterns) and comparison_df["Match"].all())
