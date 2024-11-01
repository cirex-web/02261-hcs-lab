
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
import matplotlib.patches as patches

fig1 = plt.figure(figsize=[10,10])
ax1 = fig1.add_subplot(111,aspect = 'equal')

def run_test(train_features,train_labels,test_features,test_labels):
    tree = RandomForestClassifier(n_estimators = 200).fit(train_features,train_labels)
    predictions = tree.predict_proba(test_features)
    
    test_false_positives = []
    test_false_negatives = []

    # extract out specific wrong predictions (threshold being .5)
    for i, feature in enumerate(test_features):
      predicted_label = tree.predict([feature])[0]
      actual_label = test_labels[i]
      if predicted_label != actual_label:
        if predicted_label == 1:
          test_false_positives.append(i)
        else:
          test_false_negatives.append(i)

    # print(tree.predict(test_features),predictions)
    # print("F",train_features,train_labels,test_features,test_labels,predictions)
    return test_labels,predictions[:,1],test_false_positives,test_false_negatives


# 0 is untreated, 1 is treated
 
def analyze_performance(features,labels,feature_labels):
    all_labels = []
    all_probabilities = []
    false_positives = set()
    false_negatives = set()

    for trainIndices, testIndices in KFold(n_splits=5,shuffle=True).split(features):
        actual_labels, predicted_probabilities,fp_i,fn_i = run_test([features[i] for i in trainIndices],[labels[i] for i in trainIndices],[features[i] for i in testIndices],[labels[i] for i in testIndices])
        all_labels.extend(actual_labels)
        all_probabilities.extend(predicted_probabilities)

        false_positives.update([testIndices[i] for i in fp_i])
        false_negatives.update([testIndices[i] for i in fn_i])

    fpr, tpr, t = roc_curve(all_labels, all_probabilities)
    print("actual values          ",all_labels)
    print("predicted probabilities",all_probabilities)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, alpha=1, label=f'5-fold curve w/ AUC {roc_auc}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show(block=False)

    # Train the RandomForestClassifier on the entire dataset
    rf = RandomForestClassifier(n_estimators=200).fit(features, labels)
    
    # Compute permutation feature importance
    result = permutation_importance(rf, features, labels, n_repeats=20, random_state=42, n_jobs=-1)
    # Plot feature importance
    sorted_idx = result.importances_mean.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], xerr=result.importances_std[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_labels[i] for i in sorted_idx])
    plt.xlabel("Permutation Feature Importance")
    plt.title("Feature Importance for Random Forest")
    plt.show(block=False)

    return false_positives, false_negatives
