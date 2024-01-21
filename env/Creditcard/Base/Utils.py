import random
import pandas as pd
import numpy as np
import random
from geopy.geocoders import Nominatim

# from sklearn.base import accuracy_score
from .models import *

from sklearn.metrics import precision_recall_fscore_support


def random_forest_classifer():
    train_data = File.objects.latest("id")
    file_path = train_data.file.path
    train_df = pd.read_csv(file_path)
    print(train_df.head(5))

    X_train = train_df.drop("is_fraud", axis=1)
    y_train = train_df["is_fraud"]

    def entropy(p):
        if p == 0:
            return 0
        elif p == 1:
            return 0
        else:
            return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def information_gain(left_child, right_child):
        parent = left_child + right_child
        p_parent = parent.count(1) / len(parent) if len(parent) > 0 else 0
        p_left = left_child.count(1) / len(left_child) if len(left_child) > 0 else 0
        p_right = right_child.count(1) / len(right_child) if len(right_child) > 0 else 0
        IG_p = entropy(p_parent)
        IG_l = entropy(p_left)
        IG_r = entropy(p_right)
        return (
            IG_p
            - len(left_child) / len(parent) * IG_l
            - len(right_child) / len(parent) * IG_r
        )

    def draw_bootstrap(X_train, y_train):
        bootstrap_indices = list(
            np.random.choice(range(len(y_train)), len(y_train), replace=True)
        )
        X_bootstrap = X_train.iloc[bootstrap_indices].values
        y_bootstrap = y_train.iloc[bootstrap_indices]
        return (
            X_bootstrap,
            y_bootstrap,
        )

    print(draw_bootstrap(X_train, y_train))

    def find_split_point(X_bootstrap, y_bootstrap, max_features):
        y_bootstrap = list(y_bootstrap)
        feature_ls = set()
        num_features = len(X_bootstrap[0])
        while len(feature_ls) < max_features:
            feature_idx = random.randint(0, num_features - 1)
            feature_ls.add(feature_idx)

        best_info_gain = -999
        node = None
        for feature_idx in feature_ls:
            unique_values = np.unique(X_bootstrap[:, feature_idx])

            for split_point in unique_values:
                left_child = {"X_bootstrap": [], "y_bootstrap": []}
                right_child = {"X_bootstrap": [], "y_bootstrap": []}

                for i, value in enumerate(X_bootstrap[:, feature_idx]):
                    if value <= split_point:
                        left_child["X_bootstrap"].append(X_bootstrap[i])
                        left_child["y_bootstrap"].append(y_bootstrap[i])
                    else:
                        right_child["X_bootstrap"].append(X_bootstrap[i])
                        right_child["y_bootstrap"].append(y_bootstrap[i])

                split_info_gain = information_gain(
                    left_child["y_bootstrap"], right_child["y_bootstrap"]
                )
                if split_info_gain > best_info_gain:
                    best_info_gain = split_info_gain
                    left_child["X_bootstrap"] = np.array(left_child["X_bootstrap"])
                    right_child["X_bootstrap"] = np.array(right_child["X_bootstrap"])
                    node = {
                        "information_gain": split_info_gain,
                        "left_child": left_child,
                        "right_child": right_child,
                        "split_point": split_point,
                        "feature_idx": feature_idx,
                    }

        return node

    def terminal_node(node):
        y_bootstrap = node["y_bootstrap"]
        pred = max(y_bootstrap, key=y_bootstrap.count)
        return pred

    def split_node(node, max_features, min_samples_split, max_depth, depth):
        left_child = node["left_child"]
        right_child = node["right_child"]

        del node["left_child"]
        del node["right_child"]

        if len(left_child["y_bootstrap"]) == 0 or len(right_child["y_bootstrap"]) == 0:
            empty_child = {
                "y_bootstrap": left_child["y_bootstrap"] + right_child["y_bootstrap"]
            }
            node["left_split"] = terminal_node(empty_child)
            node["right_split"] = terminal_node(empty_child)
            return

        if depth >= max_depth:
            node["left_split"] = terminal_node(left_child)
            node["right_split"] = terminal_node(right_child)
            return node

        if len(left_child["X_bootstrap"]) <= min_samples_split:
            node["left_split"] = node["right_split"] = terminal_node(left_child)
        else:
            node["left_split"] = find_split_point(
                left_child["X_bootstrap"], left_child["y_bootstrap"], max_features
            )
            split_node(
                node["left_split"], max_depth, min_samples_split, max_depth, depth + 1
            )
        if len(right_child["X_bootstrap"]) <= min_samples_split:
            node["right_split"] = node["left_split"] = terminal_node(right_child)
        else:
            node["right_split"] = find_split_point(
                right_child["X_bootstrap"], right_child["y_bootstrap"], max_features
            )
            split_node(
                node["right_split"],
                max_features,
                min_samples_split,
                max_depth,
                depth + 1,
            )

    def build_tree(
        X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features
    ):
        root_node = find_split_point(X_bootstrap, y_bootstrap, max_features)
        split_node(root_node, max_features, min_samples_split, max_depth, 1)
        return root_node

    def random_forest(
        X_train, y_train, n_estimators, max_features, max_depth, min_samples_split
    ):
        tree_ls = list()
        for i in range(n_estimators):
            X_bootstrap, y_bootstrap = draw_bootstrap(X_train, y_train)
            tree = build_tree(
                X_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split
            )
            tree_ls.append(tree)
        return tree_ls

    n_estimators = 50
    max_features = 3
    max_depth = 10
    min_samples_split = 2

    model = random_forest(
        X_train, y_train, n_estimators, max_features, max_depth, min_samples_split
    )
    print(model)
    stored_model = StoredModel_scratch.objects.create(model=model)
    return stored_model


def testing_classifier(stored_model):
    clf = stored_model.model
    if clf is not None:
        clf = stored_model.model
        test_data = TestFile.objects.latest("id")
        file_path = test_data.file.path
        test_df = pd.read_csv(file_path)
        X_test = test_df.drop("is_fraud", axis=1)
        y_test = test_df["is_fraud"]
        print(y_test)
        print(test_df.head(5))

        def predict_tree(tree, X_test):
            feature_idx = tree["feature_idx"]
            split_point = tree["split_point"]

            print("Feature Index:", feature_idx)
            print("Split Point:", split_point)
            print("Test Value:", X_test[feature_idx])

            if X_test[feature_idx] <= split_point:
                if type(tree["left_split"]) == dict:
                    return predict_tree(tree["left_split"], X_test)
                else:
                    value = tree["left_split"]
                    return value
            else:
                if type(tree["right_split"]) == dict:
                    return predict_tree(tree["right_split"], X_test)
                else:
                    return tree["right_split"]

    def predict_rf(tree_ls, X_test):
        pred_ls = list()
        for i in range(len(X_test)):
            ensemble_preds = [predict_tree(tree, X_test.values[i]) for tree in tree_ls]
            final_pred = max(ensemble_preds, key=ensemble_preds.count)
            pred_ls.append(final_pred)
        return np.array(pred_ls)

    preds = predict_rf(clf, X_test)
    print(preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average=None
    )
    # accuracy = accuracy_score(y_test, preds)
    # print("Accuracy:", accuracy)

    precision_class_0 = precision[0]
    recall_class_0 = recall[0]
    f1_class_0 = f1[0]
    precision_class_1 = precision[1]
    recall_class_1 = recall[1]
    f1_class_1 = f1[1]
    print("precision_Class_1", precision_class_1)
    print("recall class 1", recall_class_1)
    print("f1_Class_1", f1_class_1)
    analysis = analysisreport()
    # analysis.accuracy = accuracy
    analysis.f1_class_0 = f1_class_0
    analysis.precision_class_0 = precision_class_0
    analysis.recall_class_0 = recall_class_0
    analysis.precision_class_1 = precision_class_1
    analysis.recall_class_1 = recall_class_1
    analysis.f1_class_1 = f1_class_1
    analysis.save()


def calculating_distance():
    loc = Nominatim(user_agent="Geopy Library", timeout=None)
    location1 = "dhalkhu,kahtmandu,nepal"
    location2 = "jhamsikhel,lalitpur,nepal"
    getLoc1 = loc.geocode(location1)
    getLoc2 = loc.geocode(location2)
    print(getLoc1.address)
    print(getLoc2.address)
    print("Latitude 1 = ", getLoc1.latitude, "\n")
    print("Longitude 1 = ", getLoc1.longitude)
    print("Latitude 2 = ", getLoc2.latitude, "\n")
    print("Longitude 2 = ", getLoc2.longitude)
    latitudinal_distance = abs(round(getLoc1.latitude - getLoc2.latitude, 3)) / 10
    longitudinal_distance = abs(round(getLoc1.longitude - getLoc2.longitude, 3)) / 10
    print(latitudinal_distance)
    print(longitudinal_distance)
