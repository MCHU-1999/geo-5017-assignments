import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import hinge_loss
from sklearn.metrics import log_loss
from datetime import datetime

def learning_curve(X, y, classifier_type, param_set, test_sizes=np.linspace(0.1, 0.9, 9)):
    """
    Generate a learning curve by manually varying train-test splits

    Parameters:
    - X: Feature matrix
    - y: Label vector
    - classifier_type: String ('svm' or 'random_forest')
    - param_set: Dictionary of hyperparameters
    - test_sizes: Array of test set proportions
    """
    train_acc, test_acc = [], []
    train_loss, test_loss = [], []
    sizes = []
    # Create parameter string for display
    params_str = ", ".join(f"{k}={v}" for k, v in param_set.items())

    # Reverse processing order to ensure final metrics use max samples
    for test_size in sorted(test_sizes, reverse=True):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Initialize model
        if classifier_type == 'svm':
            model = svm.SVC(**param_set)
        elif classifier_type == 'random_forest':
            model = RandomForestClassifier(**param_set)
        else:
            raise ValueError("classifier_type must be 'svm' or 'random_forest'")

        model.fit(X_train, y_train)
        sizes.append(len(X_train))

        # Store metrics
        train_acc.append(accuracy_score(y_train, model.predict(X_train)))
        test_acc.append(accuracy_score(y_test, model.predict(X_test)))

        if classifier_type == 'svm':
            train_loss.append(hinge_loss(y_train, model.decision_function(X_train)))
            test_loss.append(hinge_loss(y_test, model.decision_function(X_test)))
        else:
            train_loss.append(log_loss(y_train, model.predict_proba(X_train)))
            test_loss.append(log_loss(y_test, model.predict_proba(X_test)))

        # --- CALCULATE FINAL METRICS (LAST ELEMENT) ---
    final_size = sizes[-1]
    final_train_acc = train_acc[-1]
    final_test_acc = test_acc[-1]
    final_train_loss = train_loss[-1]
    final_test_loss = test_loss[-1]
    final_gap = final_test_loss - final_train_loss

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Learning Curves ({classifier_type}): {params_str}",
                 fontsize=14, y=1.02)

    # --- Accuracy Plot ---
    ax1.plot(sizes, train_acc, 'r--o', markersize=8, label="Train Accuracy")
    ax1.plot(sizes, test_acc, 'g-s', markersize=8, label="Test Accuracy")
    ax1.set_title("Accuracy Learning Curve", fontsize=14, pad=20)
    ax1.set_xlabel("Number of Training Samples", fontsize=12)
    ax1.set_ylabel("Accuracy Score", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)

    # Annotate FINAL values
    ax1.annotate(f'Final Train: {final_train_acc:.3f}',
                 xy=(final_size, final_train_acc),
                 xytext=(10, 10), textcoords='offset points',
                 ha='left', va='bottom', fontsize=10)
    ax1.annotate(f'Final Test: {final_test_acc:.3f}',
                 xy=(final_size, final_test_acc),
                 xytext=(10, -20), textcoords='offset points',
                 ha='left', va='top', fontsize=10)

    # --- Loss Plot ---
    ax2.plot(sizes, train_loss, 'b--o', markersize=8, label="Train Loss")
    ax2.plot(sizes, test_loss, 'm-s', markersize=8, label="Test Loss")
    ax2.set_title("Loss Learning Curve", fontsize=14, pad=20)
    ax2.set_xlabel("Number of Training Samples", fontsize=12)
    ax2.set_ylabel("Loss Value", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)

    # Highlight FINAL gap
    ax2.fill_between(sizes, train_loss, test_loss, color='gray', alpha=0.2)
    ax2.annotate(f'Final Gap: {final_gap:.3f}',
                 xy=(final_size, (final_train_loss + final_test_loss) / 2),
                 xytext=(10, 0), textcoords='offset points',
                 ha='left', va='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    # Annotate FINAL values
    ax2.annotate(f'Final Train: {final_train_loss:.3f}',
                 xy=(final_size, final_train_loss),
                 xytext=(10, 10), textcoords='offset points',
                 ha='left', va='bottom', fontsize=10)
    ax2.annotate(f'Final Test: {final_test_loss:.3f}',
                 xy=(final_size, final_test_loss),
                 xytext=(10, -20), textcoords='offset points',
                 ha='left', va='top', fontsize=10)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"learning_curve_{classifier_type}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot as: {filename}")

    plt.tight_layout()
    plt.show()

    # Print numerical summary (with clear final values)
    print("\nLearning Curve Summary:")
    print(f"{'Train Size':<15}{'Train Acc':<12}{'Test Acc':<12}{'Train Loss':<12}{'Test Loss':<12}")
    for i, size in enumerate(sizes):
        print(f"{size:<15}{train_acc[i]:<12.3f}{test_acc[i]:<12.3f}"
              f"{train_loss[i]:<12.3f}{test_loss[i]:<12.3f}")

    print("\n=== FINAL METRICS ===")
    print(f"{'Train Size':<15}{'Train Acc':<12}{'Test Acc':<12}{'Train Loss':<12}{'Test Loss':<12}{'Gap':<12}")
    print(f"{final_size:<15}{final_train_acc:<12.3f}{final_test_acc:<12.3f}"
          f"{final_train_loss:<12.3f}{final_test_loss:<12.3f}{final_gap:<12.3f}")