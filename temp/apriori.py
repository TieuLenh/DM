
# MỤC 1: SỬ DỤNG THƯ VIỆN (MLXTEND) CHO THUẬT TOÁN APRIORI


import random
from itertools import combinations
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori_library(transactions, min_support=0.3, min_confidence=0.7):
    # Bước 1: Mã hóa danh sách các giao dịch (list of lists) thành ma trận True/False
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Bước 2: Chạy thuật toán Apriori của thư viện mlxtend để tìm các tập mục phổ biến (Frequent Itemsets)
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

    # Bước 3: Sinh các luật kết hợp (Association Rules) từ tập phổ biến dựa trên ngưỡng độ tin cậy (min_confidence)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Sắp xếp kết quả theo support giảm dần để dễ quan sát và đối chiếu
    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)
    if not rules.empty:
        rules = rules.sort_values(by=["confidence", "lift"], ascending=False).reset_index(drop=True)

    return frequent_itemsets, rules



# MỤC 2: TỰ VIẾT CODE THUẬT TOÁN APRIORI TỪ ĐẦU (DẠNG HÀM THƯỜNG)


def get_itemset_support(itemset, transactions_sets):
    # Tính độ hỗ trợ (support) của một tập mục (frozenset) trên toàn bộ danh sách giao dịch
    count = sum(1 for trans in transactions_sets if itemset.issubset(trans))
    return count / len(transactions_sets)


def generate_apriori_candidates(prev_frequent_itemsets, k):
    # Bước nối (Join step): tạo tập ứng viên C_k có kích thước k từ các tập phổ biến L_{k-1}
    candidates = set()
    prev_list = list(prev_frequent_itemsets)
    for i in range(len(prev_list)):
        for j in range(i + 1, len(prev_list)):
            # Hợp 2 tập phổ biến kích thước k-1 lại với nhau
            union_set = prev_list[i].union(prev_list[j])
            # Chỉ giữ lại nếu hợp của chúng có đúng k phần tử
            if len(union_set) == k:
                # Bước tỉa (Prune step - Nguyên lý Apriori):
                # Mọi tập con kích thước k-1 của tập ứng viên C_k đều phải là tập phổ biến
                all_subsets_frequent = all(
                    frozenset(sub) in prev_frequent_itemsets
                    for sub in combinations(union_set, k - 1)
                )
                if all_subsets_frequent:
                    candidates.add(union_set)
    return candidates


def custom_apriori_fit(transactions, min_support=0.3, min_confidence=0.7):
    # Chuyển mỗi giao dịch thành kiểu set để kiểm tra tập con (issubset) nhanh hơn
    transactions_sets = [set(t) for t in transactions]

    # Bước 1: Tìm tập các mục đơn lẻ duy nhất (C_1) xuất hiện trong dữ liệu
    unique_items = set(item for trans in transactions_sets for item in trans)
    c1 = [frozenset([item]) for item in unique_items]

    # Lọc ra tập phổ biến kích thước 1 (L_1) thỏa mãn >= min_support
    current_l = {}
    for itemset in c1:
        sup = get_itemset_support(itemset, transactions_sets)
        if sup >= min_support:
            current_l[itemset] = sup

    # Lưu tất cả các tập phổ biến tìm được vào dictionary frequent_itemsets
    frequent_itemsets = dict(current_l)
    k = 2

    # Bước 2: Vòng lặp tăng dần kích thước k = 2, 3, ... cho đến khi không còn tập phổ biến mới
    while current_l:
        candidates_k = generate_apriori_candidates(current_l.keys(), k)
        next_l = {}

        for candidate in candidates_k:
            sup = get_itemset_support(candidate, transactions_sets)
            if sup >= min_support:
                next_l[candidate] = sup
                frequent_itemsets[candidate] = sup

        current_l = next_l
        k += 1

    # Bước 3: Sinh các luật kết hợp (X -> Y) từ các tập phổ biến có từ 2 phần tử trở lên
    rules = []
    for itemset, support_xy in frequent_itemsets.items():
        if len(itemset) >= 2:
            for r in range(1, len(itemset)):
                for antecedent_tuple in combinations(itemset, r):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset - antecedent

                    support_x = frequent_itemsets[antecedent]
                    support_y = frequent_itemsets[consequent]

                    # Tính độ tin cậy Confidence(X -> Y) = Support(X U Y) / Support(X)
                    confidence = support_xy / support_x

                    if confidence >= min_confidence:
                        lift = confidence / support_y
                        rules.append({
                            "antecedents": antecedent,
                            "consequents": consequent,
                            "support": support_xy,
                            "confidence": confidence,
                            "lift": lift
                        })

    return frequent_itemsets, rules


def format_apriori_results(frequent_itemsets, rules):
    # Chuyển đổi kết quả tập phổ biến và luật kết hợp sang DataFrame để dễ so sánh với thư viện
    df_itemsets = pd.DataFrame([
        {"support": sup, "itemsets": itemset}
        for itemset, sup in frequent_itemsets.items()
    ])
    if not df_itemsets.empty:
        df_itemsets = df_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)

    df_rules = pd.DataFrame(rules)
    if not df_rules.empty:
        df_rules = df_rules.sort_values(by=["confidence", "lift"], ascending=False).reset_index(drop=True)

    return df_itemsets, df_rules


def run_apriori_custom(transactions, min_support=0.3, min_confidence=0.7):
    # Hàm gọi nhanh thuật toán Apriori tự viết và trả về 2 DataFrame kết quả
    frequent_itemsets, rules = custom_apriori_fit(
        transactions, min_support=min_support, min_confidence=min_confidence
    )
    return format_apriori_results(frequent_itemsets, rules)



# MỤC 3: TẠO DATA MOCK VÀ KIỂM THỬ SO SÁNH 2 THUẬT TOÁN (THƯ VIỆN VS TỰ VIẾT)


if __name__ == "__main__":
    random.seed(42)

    catalog = ["Sữa", "Bánh mì", "Bơ", "Trứng", "Cà phê", "Đường", "Táo", "Chuối"]

    mock_transactions = []
    for _ in range(200):
        basket = set()
        if random.random() < 0.55:
            basket.update(["Bánh mì", "Bơ"])
            if random.random() < 0.75:
                basket.add("Sữa")
        if random.random() < 0.45:
            basket.add("Cà phê")
            if random.random() < 0.80:
                basket.add("Đường")
        extra_items = random.sample(catalog, k=random.randint(1, 2))
        basket.update(extra_items)
        mock_transactions.append(list(basket))

    print(f"Tổng số giao dịch giả lập (Mock Transactions): {len(mock_transactions)}")
    print("Ví dụ 5 giao dịch đầu tiên:")
    for idx in range(5):
        print(f"  Giao dịch {idx + 1}: {mock_transactions[idx]}")
    print()

    MIN_SUP = 0.25
    MIN_CONF = 0.70

    # Chạy thử nghiệm với thư viện mlxtend
    df_itemsets_lib, df_rules_lib = run_apriori_library(
        mock_transactions, min_support=MIN_SUP, min_confidence=MIN_CONF
    )

    # Chạy thử nghiệm với thuật toán tự viết dạng hàm
    df_itemsets_custom, df_rules_custom = run_apriori_custom(
        mock_transactions, min_support=MIN_SUP, min_confidence=MIN_CONF
    )

    print("=== KẾT QUẢ ĐÁNH GIÁ & SO SÁNH APRIORI ===")
    print(f"Số tập phổ biến tìm được - Thư viện mlxtend : {len(df_itemsets_lib)}")
    print(f"Số tập phổ biến tìm được - Thuật toán tự viết: {len(df_itemsets_custom)}")
    print(f"Số luật kết hợp sinh ra  - Thư viện mlxtend : {len(df_rules_lib)}")
    print(f"Số luật kết hợp sinh ra  - Thuật toán tự viết: {len(df_rules_custom)}\n")

    set_itemsets_lib = set(df_itemsets_lib["itemsets"])
    set_itemsets_custom = set(df_itemsets_custom["itemsets"])
    print(f"Tập phổ biến trùng khớp 100%: {set_itemsets_lib == set_itemsets_custom}")

    set_rules_lib = set(zip(df_rules_lib["antecedents"], df_rules_lib["consequents"]))
    set_rules_custom = set(zip(df_rules_custom["antecedents"], df_rules_custom["consequents"]))
    print(f"Luật kết hợp trùng khớp 100%: {set_rules_lib == set_rules_custom}\n")

    print("Danh sách tập mục phổ biến (Frequent Itemsets - Tự viết):")
    df_itemsets_display = df_itemsets_custom.copy()
    df_itemsets_display["itemsets"] = df_itemsets_display["itemsets"].apply(lambda x: ", ".join(sorted(list(x))))
    print(df_itemsets_display.to_string(index=False))

    print("\nDanh sách luật kết hợp mạnh (Association Rules - Tự viết):")
    df_rules_display = df_rules_custom.copy()
    df_rules_display["antecedents"] = df_rules_display["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    df_rules_display["consequents"] = df_rules_display["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    print(df_rules_display[["antecedents", "consequents", "support", "confidence", "lift"]].round(4).to_string(index=False))
