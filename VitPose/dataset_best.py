import pickle
import os

TOP_K = 5

label_path = '/localdata/syb/Released_FineDiving_Dataset/Annotations/fine-grained_annotation_aqa.pkl'
train_split = '/localdata/syb/Released_FineDiving_Dataset/Annotations/train_split.pkl'
test_split = '/localdata/syb/Released_FineDiving_Dataset/Annotations/test_split.pkl'
save_dir = f"/localdata/syb/Released_FineDiving_Dataset/Annotations/top{TOP_K}_action_best_keys_train"
os.makedirs(save_dir, exist_ok=True)

# Load label data and train split
with open(label_path, 'rb') as f:
    label_data = pickle.load(f)
with open(train_split, 'rb') as f:
    train_data = pickle.load(f)
with open(test_split, 'rb') as f:
    test_data = pickle.load(f)

# Count occurrences of each action type in the whole data to get top-K
action_counts = {}
action_keys = {}
for key, value in label_data.items():
    action_type = value[0]
    if action_type in action_counts:
        action_counts[action_type] += 1
        action_keys[action_type].append(key)
    else:
        action_counts[action_type] = 1
        action_keys[action_type] = [key]

sorted_action_counts = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
top_k_action_types = [action_type for action_type, _ in sorted_action_counts[:TOP_K]]
print(f"Top {TOP_K} Action Types: {top_k_action_types}")

# For each action type, only consider keys in the training data
for action_type in top_k_action_types:
    # Filter keys to only those in train_data
    train_keys = [k for k in train_data if k in label_data and label_data[k][0] == action_type]
    if not train_keys:
        print(f"No training data for action '{action_type}'")
        best_key_list = []
    else:
        scores = [label_data[k][1] for k in train_keys]
        max_score = max(scores)
        max_keys = [k for k in train_keys if label_data[k][1] == max_score]
        best_key_list = [max_keys[0]] if max_keys else []
    file_path = os.path.join(save_dir, f"{action_type}_best_key.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(best_key_list, f)
    print(f"Saved best key for action '{action_type}' in training data to {file_path}, which is {best_key_list}, with score {max_score if best_key_list else 'N/A'}")

    test_keys = [k for k in test_data if k in label_data and label_data[k][0] == action_type]
    three_test_keys = test_keys[:3] if len(test_keys) >= 3 else test_keys
    test_file_path = os.path.join(save_dir, f"{action_type}_3test_keys.pkl")
    with open(test_file_path, "wb") as f:
        pickle.dump(three_test_keys, f)
    print(f"Saved 3 test keys for action '{action_type}' to {test_file_path}, which are {three_test_keys}, with scores {[label_data[k][1] for k in three_test_keys]}") if three_test_keys else print(f"No test keys for action '{action_type}'")
    one_test_key = [test_keys[0]] if test_keys else None
    one_test_file_path = os.path.join(save_dir, f"{action_type}_1test_key.pkl")
    with open(one_test_file_path, "wb") as f:
        pickle.dump(one_test_key, f)
    print(f"Saved 1 test key for action '{action_type}' to {one_test_file_path}, which is {one_test_key}, with score {label_data[one_test_key[0]][1] if one_test_key else 'N/A'}") if one_test_key else print(f"No test keys for action '{action_type}'")